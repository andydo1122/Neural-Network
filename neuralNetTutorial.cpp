#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>

using namespace std;

struct Connection{
    double weight;
    double deltaWeight;
};

class Neuron;

typedef vector<Neuron> Layer;

// CLASS NEURON-----------------------------------------------
class Neuron{
    public:
        Neuron(unsigned numOutputs);
    
    private:
        static double randomWeight(void) { return rand() / double(RAND_MAX);}
        double n_outputVal;
        vector<Connection> n_outputWeights;

};

Neuron::Neuron(unsigned numOutputs){
    for(unsigned c = 0; c < numOutputs; ++c){
        n_outputWeights.push_back(Connection());
        n_outputWeights.back().wight = randomWeight();
    }
}

// CLASS NET--------------------------------------------------

class Net{
    public:
        Net(const vector<unsigned> &topology);
        void feedForward(const vector<double> &inputVals) {};
        void backProp(const vector<double> &targetVals) {};
        void getResults(vector<double> &resultVals) const {};

    private:
        vector<Layer> n_layers; // n_layers[layerNum][neuronNum]

};

void Net::feedForward(const vector<double> &inputVals){
    assert(inputVals.size() == n_layers[0].size() - 1);

    // assign (latch) the input values into input neurons
    for(unsigned i = 0; i < inputVals.size(); ++i){
        n_layers[0][i].setOutputVal(inputVals[i]);
    }

    // FOrward propagate
    for( unsigned layerNum = 1; layerNum < n_layers.size(); ++layerNum){
        for(unsigned n = 0; n < n_layers[layerNum].size() - 1; ++n){
            n_layers[layerNum][n].feedForward(); 
        }
    }
}

Net::Net(const vector<unsigned> &topology){
    unsigned numLayers = topology.size();
    for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
        n_layers.push_back(Layer());
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum + 1];

        // Created a new Layer, now we need to fill it
        // with neurons, and add a bias neuron to the layer

        for(unsigned neuronNum = 0; neuronNum <= topology[layerNum]; ++neuronNum){
            n_layers.back().push_back(Neuron()); 
            cout << "Neuron is made" << endl; 
        }
    }
}

int main(){

    // e.g. (3, 2, 1)
    vector<unsigned> topology;
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);
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
