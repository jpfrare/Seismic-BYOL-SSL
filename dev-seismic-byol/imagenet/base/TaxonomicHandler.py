from pathlib import Path
from typing import Any, Tuple, Union
from scipy.io import loadmat
from collections import defaultdict, deque

#id aqui é o id do desafio
class TaxonomicHandler:
    mat: list
    son_to_fathers: defaultdict
    father_to_sons: defaultdict
    level_to_id: defaultdict
    id_to_level: defaultdict
    
    def __init__(self, mat_path):
        self.mat = loadmat(mat_path)["synsets"]
        self.son_to_fathers = None
        self.level_to_id = None
        self.id_to_level = None

        self.build_stf_fts_and_lti()
        

    from collections import defaultdict, deque

    def build_stf_fts_and_lti(self, root_id=1001):
        '''função que constroi um dicionário de filho para pais e pais para filhos, além de também construir um dicionário de nível
        para ids e id para nível. A ideia é transformar o grafo direcionado aciclico do meta.mat em uma árvore. O nível é estabelecido
        pela menor distância do índice até a raiz (entidade, id 1001).'''
        self.son_to_fathers = defaultdict(set)
        self.father_to_sons = defaultdict(set)

        for father_id in range(1, len(self.mat) + 1):
            infos = self.mat[father_id - 1][0]
            num_children = int(infos[4][0][0])

            if num_children == 0:
                continue

            for child in infos[5][0]:
                child_id = int(child)
                self.father_to_sons[father_id].add(child_id)
                self.son_to_fathers[child_id].add(father_id)

        # Calcula a menor distância a partir da raiz.
        self.id_to_level = {root_id: 0}
        self.level_to_id = defaultdict(set)
        queue = deque([root_id])

        while queue:
            node_id = queue.popleft()
            level = self.id_to_level[node_id]
            self.level_to_id[level].add(node_id)

            for child_id in self.father_to_sons[node_id]:
                if child_id not in self.id_to_level:
                    self.id_to_level[child_id] = level + 1
                    queue.append(child_id)



    def _to_coarse_classes(self, level: int, readers: list)-> int:
        id_to_label, number_of_classes = self.top_down_cut(level)

        for reader in readers:
            reader.get_id_to_label(id_to_label)

        return number_of_classes

    def get_level_descriptions(self, level):
        descriptions = []

        for challenge_id in self.level_to_id[level]:
            description = self.mat[challenge_id - 1][0][2][0]
            descriptions.append((challenge_id,description))

        return descriptions

        
