import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from Model.prototree.prototree import ProtoTree
import os

def server_project(tree: ProtoTree,
            project_loader: DataLoader,
            device,
            args: argparse.Namespace,
            log_prefix: str = 'log_projection',  # TODO
            progress_prefix: str = 'Projection'
            ) -> dict:
        
    print("Projecting prototypes to nearest training patch (without class restrictions)...")
    # Set the model to evaluation mode
    tree.eval()
    torch.cuda.empty_cache()
    # The goal is to find the latent patch that minimizes the L2 distance to each prototype
    # To do this we iterate through the train dataset and store for each prototype the closest latent patch seen so far
    # Also store info about the image that was used for projection
    global_min_proto_dist = {j: np.inf for j in range(tree.num_prototypes)}
    global_min_patches = {j: None for j in range(tree.num_prototypes)}
    global_min_info = {j: None for j in range(tree.num_prototypes)}

    # Get the shape of the prototypes
    W1, H1, D = tree.prototype_shape

    # Build a progress bar for showing the status
    projection_iter = tqdm(enumerate(project_loader),
                            total=len(project_loader),
                            desc=progress_prefix,
                            ncols=0
                            )

    
    with torch.no_grad():
        # Get a batch of data
        features, ys = next(iter(project_loader))
        batch_size = features.shape[0]
        for i, (features, ys) in projection_iter:
            features, ys = features.to(device), ys.to(device)
            # Get the features and distances
            # - features_batch: features tensor (shared by all prototypes)
            #   shape: (batch_size, D, W, H)
            # - distances_batch: distances tensor (for all prototypes)
            #   shape: (batch_size, num_prototypes, W, H)
            # - out_map: a dict mapping decision nodes to distances (indices)
            features_batch, distances_batch, out_map = tree.forward_feature_partial(features)

            # Get the features dimensions
            bs, D, W, H = features_batch.shape

            # Get a tensor containing the individual latent patches
            # Create the patches by unfolding over both the W and H dimensions
            # TODO -- support for strides in the prototype layer? (corresponds to step size here)
            patches_batch = features_batch.unfold(2, W1, 1).unfold(3, H1, 1)  # Shape: (batch_size, D, W, H, W1, H1)

            # Iterate over all decision nodes/prototypes
            for node, j in out_map.items():

                # Iterate over all items in the batch
                # Select the features/distances that are relevant to this prototype
                # - distances: distances of the prototype to the latent patches
                #   shape: (W, H)
                # - patches: latent patches
                #   shape: (D, W, H, W1, H1)
                for batch_i, (distances, patches) in enumerate(zip(distances_batch[:, j, :, :], patches_batch)):

                    # Find the index of the latent patch that is closest to the prototype
                    min_distance = distances.min()
                    min_distance_ix = distances.argmin()
                    # Use the index to get the closest latent patch
                    closest_patch = patches.view(D, W * H, W1, H1)[:, min_distance_ix, :, :]

                    # Check if the latent patch is closest for all data samples seen so far
                    if min_distance < global_min_proto_dist[j]:
                        global_min_proto_dist[j] = min_distance
                        global_min_patches[j] = closest_patch
                        global_min_info[j] = {
                            'input_image_ix': i * batch_size + batch_i,
                            'patch_ix': min_distance_ix.item(),  # Index in a flattened array of the feature map
                            'W': W,
                            'H': H,
                            'W1': W1,
                            'H1': H1,
                            'distance': min_distance.item(),
                            'nearest_input': features[batch_i].cpu().detach().numpy(),
                            # 'nearest_input': torch.unsqueeze(features[batch_i],0),
                            'node_ix': node.index,
                        }

            # Update the progress bar if required
            projection_iter.set_postfix_str(f'Batch: {i + 1}/{len(project_loader)}')

            del features_batch
            del distances_batch
            del out_map
        # Copy the patches to the prototype layer weights
        projection = torch.cat(tuple(global_min_patches[j].unsqueeze(0) for j in range(tree.num_prototypes)),
                                dim=0,
                                out=tree.prototype_layer.prototype_vectors)
        del projection
        project_info = {}
        for j in global_min_info:
            project_info[global_min_info[j]['node_ix']] = global_min_info[j]['nearest_input']
    return project_info, tree

def server_project_addon(tree: ProtoTree,
            project_loader: DataLoader,
            device,
            add_on_layers,  
            args: argparse.Namespace,
            log_prefix: str = 'log_projection',  # TODO
            progress_prefix: str = 'Projection'
            ) -> dict:
        
    print("Projecting prototypes to nearest training patch (without class restrictions)...")
    # Set the model to evaluation mode
    tree.eval()
    add_on_layers.eval()
    torch.cuda.empty_cache()
    # The goal is to find the latent patch that minimizes the L2 distance to each prototype
    # To do this we iterate through the train dataset and store for each prototype the closest latent patch seen so far
    # Also store info about the image that was used for projection
    global_min_proto_dist = {j: np.inf for j in range(tree.num_prototypes)}
    global_min_patches = {j: None for j in range(tree.num_prototypes)}
    global_min_info = {j: None for j in range(tree.num_prototypes)}

    # Get the shape of the prototypes
    W1, H1, D = tree.prototype_shape

    # Build a progress bar for showing the status
    projection_iter = tqdm(enumerate(project_loader),
                            total=len(project_loader),
                            desc=progress_prefix,
                            ncols=0
                            )

    
    with torch.no_grad():
        # Get a batch of data
        features, ys= next(iter(project_loader))
        batch_size = features.shape[0]
        for i, (features, ys) in projection_iter:
            features, ys = features.to(device), ys.to(device)
            # Get the features and distances
            # - features_batch: features tensor (shared by all prototypes)
            #   shape: (batch_size, D, W, H)
            # - distances_batch: distances tensor (for all prototypes)
            #   shape: (batch_size, num_prototypes, W, H)
            # - out_map: a dict mapping decision nodes to distances (indices)
            features = add_on_layers(features)
            features_batch, distances_batch, out_map = tree(features)

            # Get the features dimensions
            bs, D, W, H = features_batch.shape

            # Get a tensor containing the individual latent patches
            # Create the patches by unfolding over both the W and H dimensions
            # TODO -- support for strides in the prototype layer? (corresponds to step size here)
            patches_batch = features_batch.unfold(2, W1, 1).unfold(3, H1, 1)  # Shape: (batch_size, D, W, H, W1, H1)

            # Iterate over all decision nodes/prototypes
            for node, j in out_map.items():

                # Iterate over all items in the batch
                # Select the features/distances that are relevant to this prototype
                # - distances: distances of the prototype to the latent patches
                #   shape: (W, H)
                # - patches: latent patches
                #   shape: (D, W, H, W1, H1)
                for batch_i, (distances, patches) in enumerate(zip(distances_batch[:, j, :, :], patches_batch)):

                    # Find the index of the latent patch that is closest to the prototype
                    min_distance = distances.min()
                    min_distance_ix = distances.argmin()
                    # Use the index to get the closest latent patch
                    closest_patch = patches.view(D, W * H, W1, H1)[:, min_distance_ix, :, :]

                    # Check if the latent patch is closest for all data samples seen so far
                    if min_distance < global_min_proto_dist[j]:
                        global_min_proto_dist[j] = min_distance
                        global_min_patches[j] = closest_patch
                        global_min_info[j] = {
                            'input_image_ix': i * batch_size + batch_i,
                            'patch_ix': min_distance_ix.item(),  # Index in a flattened array of the feature map
                            'W': W,
                            'H': H,
                            'W1': W1,
                            'H1': H1,
                            'distance': min_distance.item(),
                            'nearest_input': features[batch_i].cpu().detach().numpy(),
                            # 'nearest_input': torch.unsqueeze(features[batch_i],0),
                            'node_ix': node.index,
                        }

            # Update the progress bar if required
            projection_iter.set_postfix_str(f'Batch: {i + 1}/{len(project_loader)}')

            del features_batch
            del distances_batch
            del out_map
        # Copy the patches to the prototype layer weights
        projection = torch.cat(tuple(global_min_patches[j].unsqueeze(0) for j in range(tree.num_prototypes)),
                                dim=0,
                                out=tree.prototype_layer.prototype_vectors)
        del projection
        project_info = {}
        for j in global_min_info:
            project_info[global_min_info[j]['node_ix']] = global_min_info[j]['nearest_input']
    return project_info, tree



def project(tree: ProtoTree,
            project_loader: DataLoader,
            device,
            args: argparse.Namespace,
            log_prefix: str = 'log_projection',  # TODO
            progress_prefix: str = 'Projection'
            ) -> dict:
        
    print("\nProjecting prototypes to nearest training patch (without class restrictions)...")
    # Set the model to evaluation mode
    tree.eval()
    torch.cuda.empty_cache()
    # The goal is to find the latent patch that minimizes the L2 distance to each prototype
    # To do this we iterate through the train dataset and store for each prototype the closest latent patch seen so far
    # Also store info about the image that was used for projection
    global_min_proto_dist = {j: np.inf for j in range(tree.num_prototypes)}
    global_min_patches = {j: None for j in range(tree.num_prototypes)}
    global_min_info = {j: None for j in range(tree.num_prototypes)}

    # Get the shape of the prototypes
    W1, H1, D = tree.prototype_shape

    # Build a progress bar for showing the status
    projection_iter = tqdm(enumerate(project_loader),
                            total=len(project_loader),
                            desc=progress_prefix,
                            ncols=0
                            )

    
    with torch.no_grad():
        # Get a batch of data
        xs, ys = next(iter(project_loader))
        batch_size = xs.shape[0]
        for i, (xs, ys) in projection_iter:
            xs, ys = xs.to(device), ys.to(device)
            # Get the features and distances
            # - features_batch: features tensor (shared by all prototypes)
            #   shape: (batch_size, D, W, H)
            # - distances_batch: distances tensor (for all prototypes)
            #   shape: (batch_size, num_prototypes, W, H)
            # - out_map: a dict mapping decision nodes to distances (indices)
            features_batch, distances_batch, out_map = tree.forward_partial(xs)

            # Get the features dimensions
            bs, D, W, H = features_batch.shape

            # Get a tensor containing the individual latent patches
            # Create the patches by unfolding over both the W and H dimensions
            # TODO -- support for strides in the prototype layer? (corresponds to step size here)
            patches_batch = features_batch.unfold(2, W1, 1).unfold(3, H1, 1)  # Shape: (batch_size, D, W, H, W1, H1)

            # Iterate over all decision nodes/prototypes
            for node, j in out_map.items():

                # Iterate over all items in the batch
                # Select the features/distances that are relevant to this prototype
                # - distances: distances of the prototype to the latent patches
                #   shape: (W, H)
                # - patches: latent patches
                #   shape: (D, W, H, W1, H1)
                for batch_i, (distances, patches) in enumerate(zip(distances_batch[:, j, :, :], patches_batch)):

                    # Find the index of the latent patch that is closest to the prototype
                    min_distance = distances.min()
                    min_distance_ix = distances.argmin()
                    # Use the index to get the closest latent patch
                    closest_patch = patches.view(D, W * H, W1, H1)[:, min_distance_ix, :, :]

                    # Check if the latent patch is closest for all data samples seen so far
                    if min_distance < global_min_proto_dist[j]:
                        global_min_proto_dist[j] = min_distance
                        global_min_patches[j] = closest_patch
                        global_min_info[j] = {
                            'input_image_ix': i * batch_size + batch_i,
                            'patch_ix': min_distance_ix.item(),  # Index in a flattened array of the feature map
                            'W': W,
                            'H': H,
                            'W1': W1,
                            'H1': H1,
                            'distance': min_distance.item(),
                            'nearest_input': torch.unsqueeze(xs[batch_i],0),
                            'node_ix': node.index,
                        }

            # Update the progress bar if required
            projection_iter.set_postfix_str(f'Batch: {i + 1}/{len(project_loader)}')

            del features_batch
            del distances_batch
            del out_map
        # Copy the patches to the prototype layer weights
        projection = torch.cat(tuple(global_min_patches[j].unsqueeze(0) for j in range(tree.num_prototypes)),
                                dim=0,
                                out=tree.prototype_layer.prototype_vectors)
        del projection

    return global_min_info, tree

def project_with_class_constraints(tree: ProtoTree,
                                    project_loader: DataLoader,
                                    device,
                                    args: argparse.Namespace,
                                    log_prefix: str = 'log_projection_with_constraints',  # TODO
                                    progress_prefix: str = 'Projection'
                                    ) -> dict:
        
    print("\nProjecting prototypes to nearest training patch (with class restrictions)...")
    # Set the model to evaluation mode
    tree.eval()
    torch.cuda.empty_cache()
    # The goal is to find the latent patch that minimizes the L2 distance to each prototype
    # To do this we iterate through the train dataset and store for each prototype the closest latent patch seen so far
    # Also store info about the image that was used for projection
    global_min_proto_dist = {j: np.inf for j in range(tree.num_prototypes)}
    global_min_patches = {j: None for j in range(tree.num_prototypes)}
    global_min_info = {j: None for j in range(tree.num_prototypes)}

    # Get the shape of the prototypes
    W1, H1, D = tree.prototype_shape

    # Build a progress bar for showing the status
    projection_iter = tqdm(enumerate(project_loader),
                            total=len(project_loader),
                            desc=progress_prefix,
                            ncols=0
                            )

    with torch.no_grad():
        # Get a batch of data
        xs, ys = next(iter(project_loader))
        batch_size = xs.shape[0]
        # For each internal node, collect the leaf labels in the subtree with this node as root. 
        # Only images from these classes can be used for projection.
        leaf_labels_subtree = dict()
        debug_leaf_dict = {}
        debug_lable_dict = {}
        debug_dis = {}
        for branch, j in tree._out_map.items():
            leaf_labels_subtree[branch.index] = set()
            for leaf in branch.leaves:
                leaf_labels_subtree[branch.index].add(torch.argmax(leaf.distribution()).item())
        # ------------- project with match class -------------
        for i, (xs, ys) in projection_iter:
            xs, ys = xs.to(device), ys.to(device)
            # Get the features and distances
            # - features_batch: features tensor (shared by all prototypes)
            #   shape: (batch_size, D, W, H)
            # - distances_batch: distances tensor (for all prototypes)
            #   shape: (batch_size, num_prototypes, W, H)
            # - out_map: a dict mapping decision nodes to distances (indices)
            features_batch, distances_batch, out_map = tree.forward_partial(xs)

            # Get the features dimensions
            bs, D, W, H = features_batch.shape

            # Get a tensor containing the individual latent patches
            # Create the patches by unfolding over both the W and H dimensions
            # TODO -- support for strides in the prototype layer? (corresponds to step size here)
            patches_batch = features_batch.unfold(2, W1, 1).unfold(3, H1, 1)  # Shape: (batch_size, D, W, H, W1, H1)

            # Iterate over all decision nodes/prototypes
            for node, j in out_map.items():
                leaf_labels = leaf_labels_subtree[node.index]
                debug_leaf_dict[j] = leaf_labels
                debug_lable_dict[j] = ys
                debug_dis[j] = distances_batch[:, j, :, :]

                # Iterate over all items in the batch
                # Select the features/distances that are relevant to this prototype
                # - distances: distances of the prototype to the latent patches
                #   shape: (W, H)
                # - patches: latent patches
                #   shape: (D, W, H, W1, H1)
                for batch_i, (distances, patches) in enumerate(zip(distances_batch[:, j, :, :], patches_batch)):
                    #Check if label of this image is in one of the leaves of the subtree
                    if ys[batch_i].item() in leaf_labels: 
                        # Find the index of the latent patch that is closest to the prototype
                        min_distance = distances.min()
                        min_distance_ix = distances.argmin()
                        # Use the index to get the closest latent patch
                        closest_patch = patches.view(D, W * H, W1, H1)[:, min_distance_ix, :, :]
                        # Check if the latent patch is closest for all data samples seen so far
                        if min_distance < global_min_proto_dist[j]:
                            global_min_proto_dist[j] = min_distance
                            global_min_patches[j] = closest_patch
                            global_min_info[j] = {
                                'input_image_ix': i * batch_size + batch_i,
                                'patch_ix': min_distance_ix.item(),  # Index in a flattened array of the feature map
                                'W': W,
                                'H': H,
                                'W1': W1,
                                'H1': H1,
                                'distance': min_distance.item(),
                                'nearest_input': torch.unsqueeze(xs[batch_i],0), # [3, 224, 224] -> [1, 3, 224, 224]
                                'node_ix': node.index,
                                'y': torch.unsqueeze(ys[batch_i],0),
                            }

            # Update the progress bar if required
            projection_iter.set_postfix_str(f'Batch: {i + 1}/{len(project_loader)}')
        # Copy the patches to the prototype layer weights
        try: 
            projection = torch.cat(tuple(global_min_patches[j].unsqueeze(0) for j in range(tree.num_prototypes)),
                    dim=0, out=tree.prototype_layer.prototype_vectors)
            del projection
            del features_batch
            del distances_batch
            del out_map

        except:
            # ------------- print the mismatch leaf and label -------------
            for j in range(tree.num_prototypes):
                if isinstance(global_min_patches[j], type(None)) :
                    words = " +++++++++++++++++++++++++++++++++ j: %s +++++++++++++++++++++++++++++++++"%str(j)
                    print(f"\033[0;30;43m{words}\033[0m")
                    print("  leaf_labels[j] :%s "%str(debug_leaf_dict[j]))
                    print("  ys[j] :%s "%str(debug_lable_dict[j]))
            # ------------- project with mismatch class if there is no match class -------------
            # Build a progress bar for showing the status
            projection_iter = tqdm(enumerate(project_loader),
                                    total=len(project_loader),
                                    desc=progress_prefix,
                                    ncols=0
                                    )

            with torch.no_grad():
                # Get a batch of data
                xs, ys = next(iter(project_loader))
                batch_size = xs.shape[0]
                # For each internal node, collect the leaf labels in the subtree with this node as root. 
                # Only images from these classes can be used for projection.
                leaf_labels_subtree = dict()
                debug_leaf_dict = {}
                debug_lable_dict = {}
                debug_dis = {}
                for branch, j in tree._out_map.items():
                    leaf_labels_subtree[branch.index] = set()
                    for leaf in branch.leaves:
                        leaf_labels_subtree[branch.index].add(torch.argmax(leaf.distribution()).item())
                
                for i, (xs, ys) in projection_iter:
                    xs, ys = xs.to(device), ys.to(device)
                    # Get the features and distances
                    # - features_batch: features tensor (shared by all prototypes)
                    #   shape: (batch_size, D, W, H)
                    # - distances_batch: distances tensor (for all prototypes)
                    #   shape: (batch_size, num_prototypes, W, H)
                    # - out_map: a dict mapping decision nodes to distances (indices)
                    features_batch, distances_batch, out_map = tree.forward_partial(xs)

                    # Get the features dimensions
                    bs, D, W, H = features_batch.shape

                    # Get a tensor containing the individual latent patches
                    # Create the patches by unfolding over both the W and H dimensions
                    # TODO -- support for strides in the prototype layer? (corresponds to step size here)
                    patches_batch = features_batch.unfold(2, W1, 1).unfold(3, H1, 1)  # Shape: (batch_size, D, W, H, W1, H1)

                    # Iterate over all decision nodes/prototypes
                    for node, j in out_map.items():
                        leaf_labels = leaf_labels_subtree[node.index]
                        debug_leaf_dict[j] = leaf_labels
                        debug_lable_dict[j] = ys
                        debug_dis[j] = distances_batch[:, j, :, :]

                        # Iterate over all items in the batch
                        # Select the features/distances that are relevant to this prototype
                        # - distances: distances of the prototype to the latent patches
                        #   shape: (W, H)
                        # - patches: latent patches
                        #   shape: (D, W, H, W1, H1)
                        for batch_i, (distances, patches) in enumerate(zip(distances_batch[:, j, :, :], patches_batch)):
                            #Check if label of this image is in one of the leaves of the subtree
                            # Find the index of the latent patch that is closest to the prototype
                            min_distance = distances.min()
                            min_distance_ix = distances.argmin()
                            # Use the index to get the closest latent patch
                            closest_patch = patches.view(D, W * H, W1, H1)[:, min_distance_ix, :, :]
                            # Check if the latent patch is closest for all data samples seen so far
                            if min_distance < global_min_proto_dist[j]:
                                global_min_proto_dist[j] = min_distance
                                global_min_patches[j] = closest_patch
                                global_min_info[j] = {
                                    'input_image_ix': i * batch_size + batch_i,
                                    'patch_ix': min_distance_ix.item(),  # Index in a flattened array of the feature map
                                    'W': W,
                                    'H': H,
                                    'W1': W1,
                                    'H1': H1,
                                    'distance': min_distance.item(),
                                    'nearest_input': torch.unsqueeze(xs[batch_i],0), # [3, 224, 224] -> [1, 3, 224, 224]
                                    'node_ix': node.index,
                                    'y': torch.unsqueeze(ys[batch_i],0),
                                }
                    # Update the progress bar if required
                    projection_iter.set_postfix_str(f'Batch: {i + 1}/{len(project_loader)}')
                # Copy the patches to the prototype layer weights
                try:
                    projection = torch.cat(tuple(global_min_patches[j].unsqueeze(0) for j in range(tree.num_prototypes)),
                            dim=0, out=tree.prototype_layer.prototype_vectors)
                    del projection
                    del features_batch
                    del distances_batch
                    del out_map

                except:
                    for j in range(tree.num_prototypes):
                        if isinstance(global_min_patches[j], type(None)) :
                            words = " +++++++++++++++++++++++++++++++++ j: %s +++++++++++++++++++++++++++++++++"%str(j)
                            print(f"\033[0;30;43m{words}\033[0m")
                        else:
                            print(" +++++++++++++++++++++++++++++++++ j: %s +++++++++++++++++++++++++++++++++"%str(j))
                        print("  leaf_labels[j] :%s "%str(debug_leaf_dict[j]))
                        print("  ys[j] :%s "%str(debug_lable_dict[j]))
                    os._exit(0) 
        del debug_leaf_dict
        del debug_lable_dict
        del debug_dis

    return global_min_info, tree
# def project_with_class_constraints(tree: ProtoTree,
#                                     project_loader: DataLoader,
#                                     device,
#                                     args: argparse.Namespace,
#                                     log: Log,  
#                                     log_prefix: str = 'log_projection_with_constraints',  # TODO
#                                     progress_prefix: str = 'Projection'
#                                     ) -> dict:
        
#     print("\nProjecting prototypes to nearest training patch (with class restrictions)...")
#     # Set the model to evaluation mode
#     tree.eval()
#     torch.cuda.empty_cache()
#     # The goal is to find the latent patch that minimizes the L2 distance to each prototype
#     # To do this we iterate through the train dataset and store for each prototype the closest latent patch seen so far
#     # Also store info about the image that was used for projection
#     global_min_proto_dist = {j: np.inf for j in range(tree.num_prototypes)}
#     global_min_patches = {j: None for j in range(tree.num_prototypes)}
#     global_min_info = {j: None for j in range(tree.num_prototypes)}

#     # Get the shape of the prototypes
#     W1, H1, D = tree.prototype_shape

#     # Build a progress bar for showing the status
#     projection_iter = tqdm(enumerate(project_loader),
#                             total=len(project_loader),
#                             desc=progress_prefix,
#                             ncols=0
#                             )

#     with torch.no_grad():
#         # Get a batch of data
#         xs, ys = next(iter(project_loader))
#         batch_size = xs.shape[0]
#         # For each internal node, collect the leaf labels in the subtree with this node as root. 
#         # Only images from these classes can be used for projection.
#         leaf_labels_subtree = dict()
        
#         for branch, j in tree._out_map.items():
#             leaf_labels_subtree[branch.index] = set()
#             for leaf in branch.leaves:
#                 leaf_labels_subtree[branch.index].add(torch.argmax(leaf.distribution()).item())
        
#         for i, (xs, ys) in projection_iter:
#             xs, ys = xs.to(device), ys.to(device)
#             # Get the features and distances
#             # - features_batch: features tensor (shared by all prototypes)
#             #   shape: (batch_size, D, W, H)
#             # - distances_batch: distances tensor (for all prototypes)
#             #   shape: (batch_size, num_prototypes, W, H)
#             # - out_map: a dict mapping decision nodes to distances (indices)
#             features_batch, distances_batch, out_map = tree.forward_partial(xs)

#             # Get the features dimensions
#             bs, D, W, H = features_batch.shape

#             # Get a tensor containing the individual latent patches
#             # Create the patches by unfolding over both the W and H dimensions
#             # TODO -- support for strides in the prototype layer? (corresponds to step size here)
#             patches_batch = features_batch.unfold(2, W1, 1).unfold(3, H1, 1)  # Shape: (batch_size, D, W, H, W1, H1)

#             # Iterate over all decision nodes/prototypes
#             for node, j in out_map.items():
#                 leaf_labels = leaf_labels_subtree[node.index]
#                 # Iterate over all items in the batch
#                 # Select the features/distances that are relevant to this prototype
#                 # - distances: distances of the prototype to the latent patches
#                 #   shape: (W, H)
#                 # - patches: latent patches
#                 #   shape: (D, W, H, W1, H1)
#                 for batch_i, (distances, patches) in enumerate(zip(distances_batch[:, j, :, :], patches_batch)):
#                     #Check if label of this image is in one of the leaves of the subtree
#                     if ys[batch_i].item() in leaf_labels: 
#                         # Find the index of the latent patch that is closest to the prototype
#                         min_distance = distances.min()
#                         min_distance_ix = distances.argmin()
#                         # Use the index to get the closest latent patch
#                         closest_patch = patches.view(D, W * H, W1, H1)[:, min_distance_ix, :, :]

#                         # Check if the latent patch is closest for all data samples seen so far
#                         if min_distance < global_min_proto_dist[j]:
#                             global_min_proto_dist[j] = min_distance
#                             global_min_patches[j] = closest_patch
#                             global_min_info[j] = {
#                                 'input_image_ix': i * batch_size + batch_i,
#                                 'patch_ix': min_distance_ix.item(),  # Index in a flattened array of the feature map
#                                 'W': W,
#                                 'H': H,
#                                 'W1': W1,
#                                 'H1': H1,
#                                 'distance': min_distance.item(),
#                                 'nearest_input': torch.unsqueeze(xs[batch_i],0),
#                                 'node_ix': node.index,
#                             }

#             # Update the progress bar if required
#             projection_iter.set_postfix_str(f'Batch: {i + 1}/{len(project_loader)}')

#             del features_batch
#             del distances_batch
#             del out_map

#         # Copy the patches to the prototype layer weights
#         projection = torch.cat(tuple(global_min_patches[j].unsqueeze(0) for j in range(tree.num_prototypes)),
#                                 dim=0, out=tree.prototype_layer.prototype_vectors)
#         del projection

#     return global_min_info, tree
