
import json
import os
import numpy as np
import torch

from train_relation_classifier.loss_utils import get_loss_criterion, get_classification_function

torch.set_printoptions(sci_mode=False)
np.set_printoptions(suppress=True, precision=4)


class TestTimeOptimization:
    def __init__(self, num_inference_steps, prompt_parser, relation_classifier, test_time_opt_config, 
                 attn_cache, tokenizer, prompt, workspace_path, device):
        self.num_inference_steps = num_inference_steps
        self.relation_classifier = relation_classifier
        self.process_config(test_time_opt_config)
        self.attn_cache = attn_cache
        self.tokenizer = tokenizer
        self.prompt = prompt
        self.workspace_path = workspace_path
        self.device = device
        
        self.relation_triplets = prompt_parser.extract_relation_triplets_from_prompt(prompt)

        self.loss_idx = 0  ## either 0 or 1, for multiple relations
        
        self.results_data = {}
        self.inference_inds = []
        self.target_map_size = 16
        if self.relation_classifier is not None:
            self.relation_classifier.eval()
            self.loss_criterion = get_loss_criterion()
            self.classification_fn = get_classification_function()
            self.inference_inds = self.prepapre_inference_indices()

            if self.relation_classifier.config is not None:
                self.target_map_size = self.relation_classifier.config.get("map_size", -1)
                if self.target_map_size < 0:
                    print(f"WARNING: target_map_size < 0")

    
    def process_config(self, config):
        self.config = config
        loss_targets_per_iteration = {
            0: 0.00001,
            1: 0.00003,
            2: 0.00005,
            3: 0.00007,
            4: 0.00007,
            5: 0.00007,
        }
        for i in range(5, self.num_inference_steps):
            loss_targets_per_iteration[i] = 0.00007
        self.config.iterative_refinement.loss_targets_per_iteration = loss_targets_per_iteration
        
    
    def prepapre_inference_indices(self):
        def assign_by_intervals(N, A):
            C = [0] * N
            prev = 0
            for a in A:
                for i in range(prev, min(a + 1, N)):
                    C[i] = a
                prev = a + 1
            return C
        
        inds = []
        if self.relation_classifier.training_config is not None:
            training_timesteps = self.relation_classifier.training_config["timesteps"]
            inds = assign_by_intervals(self.num_inference_steps, training_timesteps)
            
        return inds
            
    
    def denoise(self, latents, timestep):
        pass
    
    def extract_cross_attn_maps(self, obj1, obj2, rel_name, iter_idx):
        pass

    def get_step_size(self, iter_idx):
        return self.config.step_sizes[iter_idx]

    def get_inference_idx(self, iter_idx):
        inference_idx = iter_idx
        if len(self.inference_inds) > 0 and iter_idx < len(self.inference_inds):
            inference_idx = self.inference_inds[iter_idx]

        return inference_idx
            
        
    def prepare_relation_classifier_inputs(self, iter_idx):
        map1_list, map2_list, target_inds, iter_idx_list = [], [], [], []
        for obj1, rel_names, obj2 in [self.relation_triplets[self.loss_idx]]:
        # for obj1, rel_name, obj2 in self.relation_triplets:
            if type(rel_names) is list and len(rel_names) > 1:
                rel_name = rel_names[0]
                for new_rel_name in rel_names:
                    rel_idx = self.relation_classifier.relation_name_to_index(new_rel_name)
                    target_inds.append(rel_idx)
            else:
                rel_name = rel_names
                rel_idx = self.relation_classifier.relation_name_to_index(rel_name)
                target_inds.append(rel_idx)
                
            # map1.shape = (57 * 24, 16, 16)
            map1, map2 = self.extract_cross_attn_maps(obj1, obj2, rel_name, iter_idx)
            map1 = map1.to(torch.float32)
            map2 = map2.to(torch.float32)
            map1_list.append(map1)
            map2_list.append(map2)
            
            inference_idx = self.get_inference_idx(iter_idx)
            iter_idx_list.append(torch.tensor(inference_idx, dtype=torch.float32).unsqueeze(0))
        
        # Combine input data into batches
        # Shape: (B, 57*24, 16, 16):
        map1_batch = torch.stack(map1_list)
        map2_batch = torch.stack(map2_list)
        
        iter_idx_batch = torch.stack(iter_idx_list)  # Shape: (B, 1)
        target_inds = torch.tensor(target_inds, dtype=torch.int64)
        
        # Move to device and stack all maps into a single tensor
        map1_batch, map2_batch = map1_batch.to(self.device), map2_batch.to(self.device)
        iter_idx_batch = iter_idx_batch.to(self.device)
        target_inds = target_inds.to(self.device)
        
        # (B, num_maps, 57*24, 16, 16)
        all_maps_batches = [map1_batch, map2_batch]
        maps = torch.stack(all_maps_batches, dim=1)
        return maps, iter_idx_batch, target_inds


    def update_latents(self, latents, loss, iter_idx):
        step_size = self.get_step_size(iter_idx)
        grad = torch.autograd.grad(loss, latents, create_graph=False, retain_graph=False)[0]
        latents = latents - step_size * grad
        return latents


    def compute_loss(self, latents, timestep, iter_idx, store_results=False):
        latents, _ = self.denoise(latents, timestep)  ## To store cross-attention maps
        maps, timestep_batch, target_inds = self.prepare_relation_classifier_inputs(iter_idx)
        outputs = self.relation_classifier(maps, timestep_batch)
        loss = self.loss_criterion(outputs, target_inds)
        
        if store_results:
            predicted_classes = self.classification_fn(outputs)
            probs = torch.nn.functional.softmax(outputs, dim=1).detach().cpu().numpy()
            # print(f"{probs}")
            self.results_data[f"timestep_{iter_idx}"] = {
                "model_outputs": outputs.clone().detach().cpu().numpy().tolist(),
                "probs": probs.tolist(),
                "predicted_classes": predicted_classes.tolist(),
                "target_inds": target_inds.clone().detach().cpu().numpy().tolist(),
                "relation_triplets": self.relation_triplets,
            }
        
        return loss, latents
    
    
    def refine_latents(self, latents, timestep, iter_idx):
        """
        Continuously update the latent code according to our loss objective until the given threshold is reached,
        or the number of max refinement iteratinos is exceeded.
        """
        max_refinement_steps = self.config.iterative_refinement.max_refinement_steps[iter_idx]
        loss_targets_per_iteration = self.config.iterative_refinement.loss_targets_per_iteration
        loss_target = loss_targets_per_iteration[iter_idx] if iter_idx in loss_targets_per_iteration.keys() else 0.0
        
        success = True
        for i in range(max_refinement_steps + 1):
            self.loss_idx = i % len(self.relation_triplets)  ## cycle through relation triplets if multiple are given
            store_results = (self.config.test_classification and i == 0)
            loss, latents = self.compute_loss(latents, timestep, iter_idx, store_results)
            
            if i == 0 and loss < self.config.min_loss:
                success = False
                break
            
            if loss != 0 and loss > loss_target:
                latents = self.update_latents(latents, loss, iter_idx)
            else:
                break
                
        return success, latents
    
    
    def should_run(self, iter_idx):
        return (self.config.enable and self.config.start_idx <= iter_idx < self.config.end_idx \
            and len(self.relation_triplets) > 0)
    
    def dump_results_data(self, output_path, prefix=""):
        filename = f"{prefix}__opt_data.json" if len(prefix) > 0 else "opt_data.json"
        with open(os.path.join(output_path, filename), "w") as f:
            json.dump(self.results_data, f, indent=4)
    
    def run(self, latents, timestep, iter_idx):
        # Test-time optimization procedure:
        #  1. Run: Denoising_Model(z_t, timestep)
        #  2. For each relation triplet from (2):
        #      2.1. Extract the subject's and object's cross-attention maps: (CA1, CA2)
        #      2.2. Predict the relation name by running RelationClassifier(CA1, CA2, timestep)
        #      2.3. Compute a loss: (1.0 - the predicted relation name probability) 
        #  3. Compute the average loss.
        #  4. Perform latent code optimization on z_t accordingly.
        
        if self.should_run(iter_idx):
            input_latents = latents.clone().detach().requires_grad_(True)
            with torch.enable_grad():
                success, new_latents = self.refine_latents(latents, timestep, iter_idx)
                latents = new_latents if success else input_latents
                    
        return latents
