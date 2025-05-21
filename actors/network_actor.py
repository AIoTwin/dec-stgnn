class NetworkActor:
    def send_to(self, receiver, payload):
        receiver.on_receive(payload)