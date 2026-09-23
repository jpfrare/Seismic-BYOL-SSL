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
        y = loadmat(self.mat_path)['sysnets']
        wnid_to_description = {}

        for i in range(len(y)):
            wnid_to_description[str(y[i][0][1][0])] = str(y[i][0][3][0])

        return wnid_to_description

    def get_descriptions(self):
        return [self.wnid_to_description[wnid] for wnid in self.chosen_wnids]

    def build_son_to_father_dict(self, father_id, sysnets, dic):
        infos = sysnets[father_id - 1][0]
        num_children = int(infos[4][0][0])

        if num_children == 0:
            return
        
        for children_id in infos[5][0]:
            int_children_id = int(children_id)
            dic[int_children_id] = father_id
            self.build_son_to_father_dict(int_children_id, sysnets, dic)
        
    def build_heritage_path(self) -> dict:
        mat = loadmat(self.mat_path)
        synsets = mat['synsets']

        dic = {}
        self.build_son_to_father_dict(1001, synsets, dic)

        heritage_path = {} #wnid -> lista do wnid de todos os parentes começando por ele mesmo até a classe entidade

        for i in range(0,1000):
            imagenet_id = i + 1
            wnid = str(synsets[i][0][1][0])

            heritage_path[wnid] = [wnid]
            
            current_id = imagenet_id

            while current_id in dic:
                father_id = dic[current_id]
                father_wnid = str(synsets[father_id - 1][0][1][0])
                heritage_path[wnid].append(father_wnid)

                current_id = father_id

        
        return heritage_path


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