# 1. peer to peer averaging model (1st all-to-all with adjacency matrix)
# 2. then add different adjacency matrix where cloudts arent all connected
# 3. add parlelizm (easier to exploit super computer due to big GPUs)
# --> ray framework (multiprocessing)
# --> ray.get() WAIT FOR STUFF TO FINISH before doing other stuff????

class Cloudlet(NetworkActor):
    def __init__(self, server, clns, nodes, edges):
        pass

    def on_receive(self, sender, message):
        if isinstance(message, GlobalModelMessage):
            self.local_model = message.global_model
        else:
            assert False, 'unrecognized message type'
        
    def train_locally(self):
        ...
        #server.receive_cln_model(self, self.model)
        self.send_to(server, LocalModelMessage(self.model))

    def collect_features(self): # called every 5 minutes
        # add new local features to local training data
        for other_cln in other_clns:
            features_to_send = # compute which nodes the other cln needs
            self.send_to(other_cln, FeatureUpdateMessage(features_to_send))

class FederatedServer(NetworkActor):
    def __init__(self, clns):
        pass

    def on_receive(self, sender, message):
        if isinstance(message, LocalModelMessage):
            self.cln_models[sender] = message.model
        else:
            assert False, 'unrecognized message type'

    def average_models(self):
        self.global_model = ...
        for cln in clns:
            #cln.receive_central_model(self.global_model)
            self.send_to(cln, GlobalModelMessage(self.global_model))



class NetworkActor:
    def send_to(self, receiver, payload):
        receiver.on_receive(self, payload)

    def on_receive(self, sender, payload):
        pass


promise = foo()
....
....
await promise