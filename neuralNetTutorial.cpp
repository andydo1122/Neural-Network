#include <vector>
#include <iostream>

using namespace std;

class Net{
    public:
        Net(const vector<unsigned> &topology);
        void feedForward(const vector<double> &inputVals);
        void backProp(const vector<double> &targetVals);
        void getResults(vector<double> &resultVals) const;

    private:


};


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
