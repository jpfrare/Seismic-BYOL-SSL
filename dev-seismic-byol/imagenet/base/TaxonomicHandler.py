from pathlib import Path
from typing import Any, Tuple, Union
from scipy.io import loadmat

#id aqui é o id do desafio
class TaxonomicHandler:
    mat: list
    chosen_wnids: list
    wnid_to_description: dict
    son_to_father: dict
    level_to_id: dict


    def __init__(self, mat_path):
        self.chosen_wnids = []
        self.mat = loadmat(mat_path)["synsets"]
        self.son_to_father = {}
        self.level_to_id = {}

        self.build_stf_and_lti(1001, 0, self.son_to_father, self.level_to_id)
        


    def build_stf_and_lti(self, father_id, level, son_to_father, level_to_ids):
        infos = self.mat[father_id - 1][0]

        if level not in level_to_ids:
            level_to_ids[level] = [father_id]
        else:
            level_to_ids[level].append(father_id)

        num_children = int(infos[4][0][0])
        if num_children == 0:
            return

        for children_id in infos[5][0]:
            i_children_id = int(children_id)
            son_to_father[i_children_id] = father_id
            self.build_stf_and_lti(i_children_id, level + 1, son_to_father, level_to_ids)


    def top_down_cut(self, level: int) -> tuple[dict, int]:
        leaf_to_coarse = {}
        original_challenge_ids = range(1,1001)
        coarse_ids = set(self.level_to_id[level])

        for challenge_id in original_challenge_ids:
            father = challenge_id
            while father not in coarse_ids:
                father = self.son_to_father[father]

            leaf_to_coarse[challenge_id] = father

        new_classes = sorted(set(leaf_to_coarse.values()))
        number_of_new_classes = len(new_classes)
        new_labels = {new_classes[i] : i for i in range(number_of_new_classes)}

        id_to_label = {}
        for challenge_id in original_challenge_ids:
            coarse_id = leaf_to_coarse[challenge_id]
            new_label = new_labels[coarse_id]

            id_to_label[challenge_id] = new_label

        return (id_to_label, number_of_new_classes)

        
