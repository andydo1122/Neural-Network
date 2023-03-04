#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>

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
        Neuron(unsigned numOutputs, unsigned myIndex);
        void setOutputVal(double val){ n_outputVal = val;}
        double getOutputVal(void) const {return n_outputVal;}
        void feedForward(const Layer &prevLayer);
        void calcOutputGradients(double targetVal);
        void calcHiddenGradients(const Layer &nextLayer);
        void updateInputWeights(Layer &prevLayer);
    
    private:
        static double eta; // [0.0 ... 1.0] overall net training weight
        static double alpha; // [0.0... n] multiplier of last weight change(momentum)

        static double transferFunction(double x);
        static double transferFunctionDerivative(double x);
        static double randomWeight(void) { return rand() / double(RAND_MAX);}
        double sumDOW(const Layer &nextLayer) const; 
        double n_outputVal;
        vector<Connection> n_outputWeights;
        unsigned n_myIndex;
        double n_gradient; 

};

double Neuron::eta = 0.15; // overall net learning rate
double Neuron::alpha = 0.5; // momentum, multiplier of last deltaWeight 

void Neuron::updateInputWeights(Layer &prevLayer){
    // the weights to be updated are in the connection container
    // in the neurons in the preceding layer

    for(unsigned n = 0; n < prevLayer.size(); ++n){
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.n_outputWeights[n_myIndex].deltaWeight;

        
        double newDeltaWeight = 
            // individual input. magnified by gradient and train rate;
            eta 
            * neuron.getOutputVal()
            * n_gradient
            // Also add momentum = a fraction of the previous delta weight
            + alpha
            * oldDeltaWeight;

            neuron.n_outputWeights[n_myIndex].deltaWeight = newDeltaWeight;
            neuron.n_outputWeights[n_myIndex].weight += newDeltaWeight; 
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const{
    double sum = 0.0;

    // Sum our contributions of erros at nodes we feed
    for(unsigned n = 0; n < nextLayer.size() - 1; ++n){
        sum += n_outputWeights[n].weight * nextLayer[n].n_gradient;
    }

    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer){
    double dow = sumDOW(nextLayer);
    n_gradient = dow * Neuron::transferFunctionDerivative(n_outputVal);
}

void Neuron::calcOutputGradients(double targetVal){
    double delta = targetVal - n_outputVal;
    n_gradient = delta * Neuron::transferFunctionDerivative(n_outputVal);
}

double Neuron:: transferFunction(double x){
    // tanh - output range (-1.0..1.0)
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x){
    return 1.0 - x * x; 
}

void Neuron::feedForward(const Layer &prevLayer){
    double sum = 0.0;

    // Sum the previous layer's outputs which are our inputs...
    // Include the bias node from previous layer.

    for( unsigned n = 0; n < prevLayer.size(); ++n){
        sum += prevLayer[n].getOutputVal() *
                prevLayer[n].n_outputWeights[n_myIndex].weight;
    }

    n_outputVal = Neuron::transferFunction(sum);
}

Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
    for(unsigned c = 0; c < numOutputs; ++c){
        n_outputWeights.push_back(Connection());
        n_outputWeights.back().weight = randomWeight();
    }

    n_myIndex = myIndex;
}

// CLASS NET--------------------------------------------------

class Net{
    public:
        Net(const vector<unsigned> &topology);
        void feedForward(const vector<double> &inputVals);
        void backProp(const vector<double> &targetVals);
        void getResults(vector<double> &resultVals) const;
        double getRecentAverageError(void) const { return n_recentAverageError; }

    private:
        vector<Layer> n_layers; // n_layers[layerNum][neuronNum]
        double n_error;
        double n_recentAverageError;
        static double n_recentAverageSmoothingFactor;


};

double Net::n_recentAverageSmoothingFactor = 100.0; 

void Net::getResults(vector<double> &resultVals) const {
    resultVals.clear();

    for(unsigned n = 0; n < n_layers.back().size() - 1; ++n){
        resultVals.push_back(n_layers.back()[n].getOutputVal());
    }
}

void Net::backProp(const vector<double> &targetVals){
    // Calculate overall net error(RMS/ROOT MEAN SQUARE of output neuron errors)
    Layer &outputLayer = n_layers.back();
    n_error = 0.0;

    for(unsigned n = 0; n < outputLayer.size() - 1; ++n){
        double delta = targetVals[n] - outputLayer[n].getOutputVal();
        n_error += delta*delta;
    }

    n_error /= outputLayer.size() - 1; // get average error squared
    n_error = sqrt(n_error); // RMS

    // Implement recent average measurement;
    n_recentAverageError = (n_recentAverageError* n_recentAverageSmoothingFactor + n_error)
                            / (n_recentAverageSmoothingFactor + 1.0);

    // calculate output layer gradients
    for(unsigned n = 0; n < outputLayer.size() - 1; ++n){
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    // calculate gradients on hidden layers
    for(unsigned layerNum = n_layers.size() - 2; layerNum> 0; --layerNum){
        Layer &hiddenLayer = n_layers[layerNum];
        Layer &nextLayer = n_layers[layerNum + 1];

        for(unsigned n = 0; n < hiddenLayer.size(); ++n){
            hiddenLayer[n].calcHiddenGradients(nextLayer);
        }
    }

    // for all layers from outputs to first hidden layer.
    // update conncection weights 
    for(unsigned layerNum = n_layers.size() - 1; layerNum > 0; --layerNum){
        Layer &layer = n_layers[layerNum];
        Layer &prevLayer = n_layers[layerNum - 1];

        for(unsigned n = 0; n < layer.size() - 1; ++n){
            layer[n].updateInputWeights(prevLayer);
        }

    }

}

void Net::feedForward(const vector<double> &inputVals){
    cout << inputVals.size() << " " << n_layers[0].size() - 1 << endl;
    assert(inputVals.size() == n_layers[0].size() - 1);
    cout << "Neuron is fed" << endl;
    // assign (latch) the input values into input neurons
    for(unsigned i = 0; i < inputVals.size(); ++i){
        n_layers[0][i].setOutputVal(inputVals[i]);
    }

    // FOrward propagate
    for( unsigned layerNum = 1; layerNum < n_layers.size(); ++layerNum){
        Layer &prevLayer = n_layers[layerNum - 1];
        for(unsigned n = 0; n < n_layers[layerNum].size() - 1; ++n){
            n_layers[layerNum][n].feedForward(prevLayer); 
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
            n_layers.back().push_back(Neuron(numOutputs, neuronNum)); 
            cout << "Neuron is made" << endl; 
        }

        // force the bias node's output values to 1.0. 
        n_layers.back().back().setOutputVal(1.0); 
    }
}

int main(){

    // e.g. (3, 2, 1)
    vector<unsigned> topology;
    topology.push_back(2);
    topology.push_back(4);
    topology.push_back(1);
    Net myNet(topology);
    bool pass = false;
    int count = 0;
    vector<double> inputVals, targetVals, resultVals; 
    while(!pass){
        //vector<double> inputVals;
        int n1 = (int) (2.0 * rand() / double(RAND_MAX));
        int n2 = (int) (2.0 * rand() / double(RAND_MAX));
        double n3 = n1;
        double n4 = n2; 
        inputVals.clear();
        inputVals.push_back(n3);
        inputVals.push_back(n4);
        myNet.feedForward(inputVals);
        cout << "INPUT VALUE----" << n3 << " " << n4 << endl;

        /*// During training
        cout << "Here at training" << endl;
        vector<double> targetVals;
        double target = (double) (n1^n2);
        targetVals.push_back( target ); 
        myNet.backProp(targetVals);
        cout << "BACK_PROP: ----" << endl;
        for(int i = 0; i < targetVals.size(); ++i){
            cout << targetVals[i] << endl;
        }*/

        // Neural network's outputs
        double target = (double) (n1 | n2);
        cout << "TARGET: " << target << endl;
        //vector<double> resultVals;
        myNet.getResults(resultVals);
        cout << "RESULTS: ----" << endl;
        for(int i = 0; i < resultVals.size(); ++i){
            cout << resultVals[i] << endl;
            if(resultVals[i] == target){
                cout << "PASS" << endl;
                pass = true; 
            }
        }

        // During training
        //cout << "Here at training" << endl;
        //vector<double> targetVals;
        targetVals.clear(); 
        targetVals.push_back( target ); 
        myNet.backProp(targetVals);
        cout << "BACK_PROP: ----" << endl;
        //for(int i = 0; i < targetVals.size(); ++i){
        //    cout << targetVals[i] << endl;
        //}

        cout << "NET RECENT AVERAGE ERROR: " << myNet.getRecentAverageError() << endl;

        cout << "COUNT: " << count++ << endl; 
    }

    return 0;
}
