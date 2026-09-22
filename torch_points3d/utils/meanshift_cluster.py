import numpy as np
import torch
import torch
from sklearn.cluster import MeanShift
import time
import multiprocessing
from multiprocessing import Process
from functools import partial
from torch_points3d.utils.meanshift_gpu import mean_shift as mean_shift_gpu


def meanshift_cluster(prediction, bandwidth):
    """Flat-kernel mean shift with bin seeding; returns int64 labels as a CPU tensor.

    On a CUDA tensor the batched torch implementation runs on the GPU (same labels as
    scikit-learn, about 40x faster per training batch); otherwise scikit-learn is used.
    """
    if torch.is_tensor(prediction) and prediction.is_cuda:
        return mean_shift_gpu(prediction, bandwidth)
    if torch.is_tensor(prediction):
        prediction = prediction.cpu().numpy()
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(prediction)
    return torch.from_numpy(ms.labels_)
        
def cluster_loop(embed_logits_logits_u, unique_in_batch, label_batch, local_ind, low, high, loop_num):
    
    #t = time.time()
    all_clusters = []
    cluster_type = []
    final_result = []
    local_logits = []
    cluster_type_loop = []
    
    embed_logits_logits_u = embed_logits_logits_u.cpu().detach()
    unique_in_batch = unique_in_batch.cpu().detach()
    label_batch = label_batch.cpu().detach()
    local_ind = local_ind.cpu().detach()
    pick_num  = 5 #np.random.randint(low=low,high=high+1,size=loop_num)
    for loop_i in range(loop_num):
        #feature_choose = np.random.choice(embed_logits_logits_u.shape[-1], pick_num[loop_i], replace=False)
        #feature_choose = torch.multinomial(torch.ones(embed_logits_logits_u.shape[-1]), pick_num[loop_i], replacement=False, out=None)
        feature_choose = torch.multinomial(torch.ones(embed_logits_logits_u.shape[-1]), pick_num, replacement=False, out=None)
        #print('type %d feature choose:' % (loop_i))
        #print(feature_choose)
        embed_logits_logits_typei = embed_logits_logits_u[:,feature_choose]
        for s in unique_in_batch:
            batch_mask = label_batch == s
            if torch.sum(batch_mask)>5:
                sampleInBatch_local_ind = local_ind[batch_mask]
                local_logits.append(sampleInBatch_local_ind)
                sample_embed_logits = embed_logits_logits_typei[batch_mask]
                #sample_embed_logits = torch.nn.functional.normalize(sample_embed_logits, dim=0)
                all_clusters.append(sample_embed_logits.cpu().detach().numpy())
                cluster_type_loop.append(loop_i)
    
    if unique_in_batch.shape[0]>0:
        processes=unique_in_batch.shape[0]
    else:
        processes=1
    with multiprocessing.Pool(processes=processes) as pool:
        results = pool.map(meanshift_cluster, all_clusters)
        for i in range(len(results)):
            pre_ins_labels_embed = results[i]
            sampleInBatch_local_ind = local_logits[i]
            loop_i_ = cluster_type_loop[i]
            unique_preInslabels = torch.unique(pre_ins_labels_embed)
            for l in unique_preInslabels:
                if l == -1:
                    continue
                label_mask_l = pre_ins_labels_embed == l
                final_result.append(sampleInBatch_local_ind[label_mask_l])
                cluster_type.append(loop_i_)
    
    #print("total time",time.time()-t)
    return final_result, cluster_type

def cluster_single(embed_logits_logits_u, unique_in_batch, label_batch, local_ind, type, bandwidth):
    #t = time.time()
    all_clusters = []
    cluster_type = []
    final_result = []
    local_logits = []

    # embeddings stay on their device (GPU mean shift); the index bookkeeping is done on the CPU
    embed_logits_logits_u = embed_logits_logits_u.detach()
    unique_in_batch = unique_in_batch.cpu().detach()
    label_batch = label_batch.cpu().detach()
    local_ind = local_ind.cpu().detach()

    for s in unique_in_batch:
        batch_mask = label_batch == s
        if torch.sum(batch_mask)>3:
            sampleInBatch_local_ind = local_ind[batch_mask]
            local_logits.append(sampleInBatch_local_ind)
            sample_embed_logits = embed_logits_logits_u[batch_mask.to(embed_logits_logits_u.device)]
            all_clusters.append(sample_embed_logits)
    
    partial_meanshift_cluster = partial(meanshift_cluster, bandwidth=bandwidth)
    results = [partial_meanshift_cluster(cluster) for cluster in all_clusters]
    for i in range(len(results)):
        pre_ins_labels_embed = results[i]
        sampleInBatch_local_ind = local_logits[i]
        unique_preInslabels = torch.unique(pre_ins_labels_embed)
        for l in unique_preInslabels:
            if l == -1:
                continue
            label_mask_l = pre_ins_labels_embed == l
            final_result.append(sampleInBatch_local_ind[label_mask_l])
            cluster_type.append(type)        
            
            #pre_ins_labels_embed = hdbscan_cluster(sample_embed_logits)
            #unique_preInslabels = torch.unique(pre_ins_labels_embed)
            #for l in unique_preInslabels:
            #    if l == -1:
            #        continue
            #    label_mask_l = pre_ins_labels_embed == l
            #    all_clusters.append(sampleInBatch_local_ind[label_mask_l])
            #    cluster_type.append(type)
                
    #print("total time",time.time()-t)
    return final_result, cluster_type

if __name__ == "__main__":
    embed_logits_logits_u = torch.load("embed_logits_logits_u.pt").cpu()
    label_batch = torch.load("label_batch.pt").cpu()
    local_ind = torch.load("local_ind.pt").cpu()
    unique_in_batch = torch.load("unique_in_batch.pt").cpu()
    print(np.shape(embed_logits_logits_u))
    print(np.shape(label_batch))
    print(np.shape(local_ind))
    print(np.shape(unique_in_batch))
    t = time.time()
    for i in range(4):
        cluster_loop(embed_logits_logits_u, unique_in_batch, label_batch, local_ind, low=3, high=5, loop_num=10)
    print("total time",time.time()-t)