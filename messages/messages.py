class MasterModelMessage():
    def __init__(self, master_model):
        self.master_model = master_model

class CloudletModelMessage():
    def __init__(self, cln_model, cln_id):
        self.cln_model = cln_model
        self.cln_id = cln_id

class GetCloudletModelRequestMessage():
    def __init__(self, cln_adj_matrix, cln_actors):
        self.cln_adj_matrix = cln_adj_matrix
        self.cln_actors = cln_actors