# -*- coding:utf-8 -*-


# class for
# 1. number of poison
# 2. indices to be poisoned
class Poison:
    def __init__(self, num, indices, backdoor_type, sources, targets, percent_poison):
        self.num_poison = num
        self.indices_to_be_poisoned = indices
        self.backdoor_type = backdoor_type
        self.sources = sources
        self.targets = targets
        self.percent_poison = percent_poison
        self.random_selection_indices = None
        self.shuffled_indices = None

    def get_num_poison(self):
        return self.num_poison

    def set_num_poison(self, num):
        self.num_poison = num

    def get_indices_to_be_poisoned(self):
        return self.indices_to_be_poisoned

    def set_indices_to_be_poisoned(self, indices):
        self.indices_to_be_poisoned = indices

    def get_backdoor_type(self):
        return self.backdoor_type

    def set_backdoor_type(self, backdoor_type):
        self.backdoor_type = backdoor_type

    def get_sources(self):
        return self.sources

    def set_sources(self, sources):
        self.sources = sources

    def get_targets(self):
        return self.targets

    def set_targets(self, targets):
        self.targets = targets

    def get_percent_poison(self):
        return self.percent_poison

    def set_percent_poison(self, percent_poison):
        self.percent_poison = percent_poison

    def get_random_selection_indices(self):
        return self.random_selection_indices

    def set_random_selection_indices(self, indices):
        self.random_selection_indices = indices

    def get_shuffled_indices(self):
        return self.shuffled_indices

    def set_shuffled_indices(self, indices):
        self.shuffled_indices = indices
