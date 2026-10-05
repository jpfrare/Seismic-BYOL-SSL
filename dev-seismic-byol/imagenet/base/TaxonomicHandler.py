from pathlib import Path
from typing import Any, Tuple, Union
from scipy.io import loadmat

class TaxonomicHandler:
    mat_path: Path
    chosen_wnids: list
    wnid_to_description: dict


    def __init__(self, mat_path):
        self.chosen_wnids = []
        self.mat_path = mat_path
        self.wnid_to_description = self.build_wnid_to_description()


    def build_wnid_to_description(self):
        y = loadmat(self.mat_path)['synsets']
        wnid_to_description = {}

        for i in range(len(y)):
            wnid_to_description[str(y[i][0][1][0])] = str(y[i][0][2][0])

        return wnid_to_description


    def build_stf_and_lti(self, father_id, level, son_to_father, level_to_ids, synsets):
        infos = synsets[father_id - 1][0]

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
            self.build_stf_and_lti(i_children_id, level + 1, son_to_father, level_to_ids, synsets)


    def reduce_taxonomic_diversity(self, wnids: list, top_down: bool, level: int) -> tuple[dict, int]:
        #wnids: lista com todos os wnids dos quais se deseja generalizar em determinado nivel
        wnid_to_class = {}
        coarse_to_class = {}
        self.chosen_wnids = []
        current_class_id = 0

        heritage_path = self.build_heritage_path()

        for wnid in wnids:
            ancestors = heritage_path[wnid] #lista de todos os wnids antepassados até entidade (o último elemento é a classe entidade)

            if top_down:
                pos = len(ancestors) - 1 - level
                chosen_wnid = ancestors[pos] if pos > 0 else ancestors[0]
            else:
                chosen_wnid = ancestors[level] if level < len(ancestors) else ancestors[3]

            if chosen_wnid not in self.chosen_wnids:
                self.chosen_wnids.append(chosen_wnid)

            if chosen_wnid not in coarse_to_class:
                coarse_to_class[chosen_wnid] = current_class_id
                current_class_id += 1
            
            wnid_to_class[wnid] = coarse_to_class[chosen_wnid]

        return (wnid_to_class, current_class_id)