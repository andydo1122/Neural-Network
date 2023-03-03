class Net{
    public:


    private:


};


int main(){

    Net myNet(topology);

    myNet.feedForward(inputVals);

    // During training
    myNet.backProp(targetVals);

    // Neural network's outputs
    myNet.getResults(resultVals);
}
