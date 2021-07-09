import torch
def custom_collate_fn(batch):
    """
    Collate the list of dictionnaries into a list of batched dictionnaries
    """

    batch_dict = dict.fromkeys(batch[0].keys())
    batch_dict['id'] = [b['id'] for b in batch]
    batch_dict['t'] = [b['t'] for b in batch]
    batch_dict['obs'] = [b['obs'] for b in batch]
    #batch_dict['obs'] = torch.cat([b['obs'] for b in batch])
    # batch_dict['tstar'] = [torch.FloatTensor(b['tstar']).view(-1, 1) for b in batch]  # Float by default
    # batch_dict['age'] = [torch.FloatTensor(b['age']).view(-1, 1) for b in batch]  # Float by default
    # batch_dict['observations'] = [torch.stack(b['observations'], 0) for b in batch]  # Float by default

    # TODO Raphael : check si on peut mettre les cofacteurs ici ? (cf tâche annexe ?) sinon au pire en dataframe dans le trainer

    # Cofactors
    batch_dict['cofactors'] = {}
    for cofactor in batch[0]["cofactors"].keys():
        batch_dict['cofactors'][cofactor] = [b["cofactors"][cofactor] for b in batch]

    # Cofactors
    batch_dict['time_label'] = {}
    for time_label in batch[0]["time_label"].keys():
        batch_dict['time_label'][time_label] = [b["time_label"][time_label] for b in batch]

    # Get index of visits of each patient
    idx_pa = []
    current_idx = 0
    for t in batch_dict['t']:
        pos_patient = list(range(current_idx, current_idx+len(t)))
        idx_pa.append(pos_patient)
        current_idx = current_idx+len(t)
    batch_dict['idx_pa'] = idx_pa

    return batch_dict