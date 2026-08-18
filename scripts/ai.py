

class Module:
    def __call__(self,x):
        return self.forward(x)
    
    def forward(self,x):
        raise NotImplementedError(
            "Subclasses must implement forward()."
            )
    
s = Module()
s.forward(1)


class Linear(Module):
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = None
        self.bias = None

        self.input = None

    def forward(self, x):
        pass