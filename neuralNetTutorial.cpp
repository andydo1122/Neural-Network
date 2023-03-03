#include <vector>
#include <iostream>

using namespace std;

class Neuron;
typedef vector<Neuron> Layer;

class Net{
    public:
        Net(const vector<unsigned> &topology);
        void feedForward(const vector<double> &inputVals) {};
        void backProp(const vector<double> &targetVals) {};
        void getResults(vector<double> &resultVals) const {};

    private:
        vector<Layer> n_layers; // n_layers[layerNum][neuronNum]

};

Net::Net(const vector<unsigned> &topology){
    unsigned numLayers = topology.size();
    for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
        n_layers.push_back(Layer());

        // Created a new Layer, now we need to fill it
        // with neurons, and add a bias neuron to the layer

        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
            n_layers.back().push_back(Neuron()); 
        }
    }
}

int main(){

    // e.g. (3, 2, 1)
    vector<unsigned> topology;
    Net myNet(topology);
    
    vector<double> inputVals;
    myNet.feedForward(inputVals);

    // During training
    vector<double> targetVals;
    myNet.backProp(targetVals);

    // Neural network's outputs
    vector<double> resultVals;
    myNet.getResults(resultVals);
}
