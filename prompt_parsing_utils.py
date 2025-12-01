import re
from train_relation_classifier.data_creation import defs as data_defs


# Maps short/non-canonical forms to canonical spatial relations
SPATIAL_RELATION_NORMALIZATION_MAP = {
    "left": {
        "left of": "to the left of",
        "on the left of": "to the left of",
    },
    "right": {
        "right of": "to the right of",
        "on the right of": "to the right of",
    },
    "top": {
        "top": "above",
        "on top of": "above",
        "on the top of": "above",
    },
    "bottom": {
        "bottom": "below",
        "on bottom of": "below",
        "on the bottom of": "below",
    },
}

DIAGONAL_PREPOSITION = " and to the "


class PromptParser:
    def __init__(self, spatial_relations_list=data_defs.SPATIAL_RELATIONS_LIST, skip_attrs=True):
        import spacy
        self.nlp_parser = spacy.load("en_core_web_sm")

        spatial_relations_list = sorted(spatial_relations_list, key=len, reverse=True)
        self.spatial_rel_tokens = [rel.split() for rel in spatial_relations_list]
        self.skip_attrs = skip_attrs


    def extract_attribute_object_pairs(self, text):
        doc = self.nlp_parser(text)
        pairs = []

        for token in doc:
            # Check if the token is an adjective (e.g. 'white', 'black')
            if token.pos_ == "ADJ":
                # Look for the noun it modifies
                for child in token.children:
                    if child.dep_ in ("amod", "acomp") and child.pos_ == "NOUN":
                        pairs.append((token.text, child.text))
                # Alternatively, check the head if the adjective is a modifier
                if token.dep_ == "amod" and token.head.pos_ == "NOUN":
                    pair = (token.head.text, data_defs.RELATION_NAME_IS, token.text)
                    pairs.append(pair)

        return pairs


    def clean_prompt_start(self, prompt):
        # Match common intro phrases (extend as needed)
        pattern = r"^(a|an)\s+(photo|drawing|illustration|painting|hyper-realistic digital painting|cinematic sci-fi movie scene|3d animation|top-down view)\s+of\s+"
        return re.sub(pattern, "", prompt.strip(), flags=re.IGNORECASE)
    
    
    def normalize_prompt_spatial_relations(self, prompt):
        for keyword, short_to_canonical_dict in SPATIAL_RELATION_NORMALIZATION_MAP.items():
            if keyword not in prompt:
                continue
            
            max_short_form = ""
            for short_form in short_to_canonical_dict.keys():
                if short_form in prompt and len(short_form) > len(max_short_form):
                    max_short_form = short_form
            
            if len(max_short_form) > 0:
                canonical_form = short_to_canonical_dict[max_short_form]
                if canonical_form in prompt:
                    continue
                pattern = re.compile(rf"\b{re.escape(max_short_form)}\b", re.IGNORECASE)
                prompt = pattern.sub(canonical_form, prompt)
            
        return prompt
    
    
    def remove_is_or_was(self, prompt):
        return re.sub(r'\b(is|was|were|are)\b', '', prompt, flags=re.IGNORECASE).replace("  ", " ").strip()
    
    def handle_exception_noun_phrases(self, prompt, noun_phrase):
        output_phrase = noun_phrase
        exceptions = {"fire hydrant", "hot dog", "potted plant", "teddy bear", "furby", "jigglypuff", 
                      "margot robbie", "stop sign"}
        for exception in exceptions:
            if exception in prompt and noun_phrase != exception and noun_phrase in exception:
                output_phrase = exception
        
        return output_phrase
    
    
    def handle_attribute_exceptions(self, chunk, noun_root):
        triplets = []
        if "orange" in chunk.text:
            # Find the main noun (head) in the chunk
            head = None
            for token in chunk:
                if "fire hydrant" in chunk.text or "hot dog" in chunk.text:
                    if token.text == "fire":
                        head = token
                        break
                elif token.pos_ == "NOUN":
                    head = token
            if not head:
                return []

            # Reconstruct full compound noun for head
            attr_token = None
            for t in head.lefts:
                if t.text == "orange" and (t.dep_ == "compound" or t.dep_ == "amod"):
                    attr_token = t.text
            
            if attr_token is not None:
                if "orange" in noun_root:
                    obj1_root = re.sub(r'^\w+\s+', '', noun_root)
                else:
                    obj1_root = noun_root
                triplets.append((obj1_root, data_defs.RELATION_NAME_IS, attr_token))
                
        elif "hot dog" in chunk.text:
            head = None
            for token in chunk:
                if token.pos_ == "NOUN":
                    head = token
            if not head:
                return []
            
            for token in chunk:
                if token.dep_ == "amod" and token.head in chunk and token.text != "hot":
                    triplets.append(("hot dog", data_defs.RELATION_NAME_IS, token.text))
        
        return triplets
    
    
    def handle_diagonals(self, prompt, triplets):
        def is_incorrect(obj):
            return (obj == "left" or obj == "right" or obj == "above" or obj == "below")
        
        
        output_triplets = []
        if len(triplets) == 0:
            return []
        
        for diag_rel in data_defs.SPATIAL_RELATION_DIAGONALS_HYPHEN:
            if diag_rel in prompt:
                for obj1, _, obj2 in triplets:
                    if is_incorrect(obj1) or is_incorrect(obj2):
                        continue
                    vert, horiz = diag_rel.split("-")
                    # horiz = horiz.split(" of")[0]
                    horiz = horiz.split("to the ")[1].split(" of")[0]
                    encoded_rel = f"{vert}_{horiz}"
                    output_triplets.append((obj1, encoded_rel, obj2))
                    break
        
        if len(output_triplets) == 0:
            output_triplets = triplets
        
        return output_triplets
    
    
    def extract_relation_triplets_from_prompt(self, prompt):
        prompt = self.clean_prompt_start(prompt)
        prompt = self.normalize_prompt_spatial_relations(prompt)
        prompt = self.remove_is_or_was(prompt)
        
        doc = self.nlp_parser(prompt)
        triplets = []

        def get_noun_root(chunk):
            return " ".join([t.text for t in chunk if t.pos_ in ("NOUN", "PROPN")])

        # Heuristic: grab previous word if it's a NOUN or ADJ (for compounds like "cell phone", "fire hydrant")
        def find_compound_or_adjacent_noun(token):
            toks = []
            # Look back up to 2 tokens (change the range for longer compounds if needed)
            idx = token.i
            # while idx > 0 and (doc[idx - 1].pos_ in ("NOUN", "ADJ", "PROPN") or doc[idx - 1].dep_ == "compound"):
            while idx > 0 and (doc[idx - 1].pos_ in ("NOUN", "ADJ") or doc[idx - 1].dep_ == "compound"):
                toks.insert(0, doc[idx - 1].text)
                idx -= 1
            toks.append(token.text)
            return " ".join(toks)
        
        # Find the nearest noun to the left of the relation span
        def find_left_noun(span):
            for tok in reversed(doc[:span.start]):
                if tok.pos_ == "NOUN":
                    return tok
            # Fallback: return the head (original logic)
            return span[0].head

        # (1) Attribute bindings
        for chunk in doc.noun_chunks:
            if self.skip_attrs:
                continue
            
            noun = get_noun_root(chunk)
            # Optional: skip known multi-word noun phrases (edit as needed)
            if "hot dog" in chunk.text or "potted plant" in chunk.text or "orange" in chunk.text:
                attr_triplet = self.handle_attribute_exceptions(chunk, noun)
                if len(attr_triplet) > 0:
                    triplets += attr_triplet
                continue
            
            for token in chunk:
                # if token.dep_ == "amod" and token.head in chunk:
                if token.pos_ == "ADJ" and token.head in chunk:
                    triplets.append((noun, data_defs.RELATION_NAME_IS, token.text))

        # (2) Spatial relations
        for i in range(len(doc)):
            for rel in self.spatial_rel_tokens:
                rel_len = len(rel)
                if i + rel_len > len(doc):
                    continue
                span = doc[i:i+rel_len]
                if span.text.lower() != " ".join(rel):
                    continue

                # Find object2
                last_token = span[-1]
                object2 = next((c for c in last_token.children if c.dep_ == "pobj"), None)
                if not object2:
                    continue

                obj2_chunk = next((c for c in doc.noun_chunks if object2 in c), None)
                if obj2_chunk:
                    obj2 = get_noun_root(obj2_chunk)
                else:
                    obj2 = find_compound_or_adjacent_noun(object2)

                # Find object1 (head of first prep word)
                # object1 = span[0].head
                object1 = find_left_noun(span)
                
                obj1_chunk = next((c for c in doc.noun_chunks if object1 in c), None)
                if obj1_chunk:
                    obj1 = get_noun_root(obj1_chunk)
                else:
                    obj1 = find_compound_or_adjacent_noun(object1)

                obj1 = self.handle_exception_noun_phrases(prompt, obj1)
                obj2 = self.handle_exception_noun_phrases(prompt, obj2)
                triplets.append((obj1, span.text, obj2))

        triplets = self.handle_diagonals(prompt, triplets)
        return triplets



if __name__ == "__main__":
    # prompt = "a photo of an orange traffic light to the right of a white toilet"
    # prompt = "a photo of an orange traffic light on the right of a white toilet"
    prompt = "a photo of a black stop sign"
    # prompt = "a photo of a blue pizza and a yellow baseball glove"
    # prompt = "a photo of a blue pizza above a yellow baseball glove"
    prompt = "a photo of a book above-left of a laptop"
    parser = PromptParser(skip_attrs=False)
    print(parser.extract_relation_triplets_from_prompt(prompt))
