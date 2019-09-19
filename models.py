class Sequential:
    def __init__(self):
        self.trainable = True #模型是否可训练
        self.layers = []
    def add(self,layer):
        self.layers.append(layer)
    #编译模型
    def compile(self,loss, optimizer,metrics):
        output_shape = None
        for one in self.layers:
            output_shape = one.init(output_shape)

        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    #训练模型
    def fit(self):
        pass
    #评估,返回评分scores
    def evaluate(self):
        pass
    # 预测，返回预测结果
    def predict(self):
        pass
    def save(self):
        pass
    def load(self):
        pass




