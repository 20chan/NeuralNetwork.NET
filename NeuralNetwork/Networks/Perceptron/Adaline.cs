using System;
using System.Linq;
using NeuralNetwork.Networks.Trains;

namespace NeuralNetwork.Networks.Perceptron
{
    public class Adaline
    {
        private int _layer;
        public int Layer { get { return _layer; } }
        private int _epoch;
        public int Epoch { get { return _epoch; } }
        private double _error;
        public double Error { get { return _error; } }

        private double[] _weights;
        private double _biasWeight;
        
        public Adaline(int layer)
        {
            _layer = layer;
            _weights = new double[layer];
            Reset();
        }

        public void Reset()
        {
            Random r = new Random();
            for (int i = 0; i < _layer; i++)
                _weights[i] = r.NextDouble() - 0.5;
            _biasWeight = r.NextDouble() - 0.5;
            _error = 1;
        }

        public void Train(BasicTrainSet<double> trainset, double learnRate = 0.3)
        {
            _error = 0;
            _epoch++;
            for (int p = 0; p < trainset.DataCount; p++)
            {
                double result = Compute(trainset.Input[p], false);
                double error = derivative(result) * (trainset.Output[p] - result);

                for (int i = 0; i < _weights.Length; i++)
                {
                    _weights[i] += error * trainset.Input[p][i] * learnRate;
                }
                _biasWeight += error * learnRate;
                _error += Math.Abs(error);
            }
            if(_epoch % 100 == 0) Console.WriteLine($"epoch : {Epoch} error : {_error}");
        }

        public double Compute(double[] input, bool quan)
        {
            double result = sigmoid(_weights.Zip(input, (a, b) => (a * b)).Sum() + _biasWeight);
            return quan ? (result >= 0.5 ? 1 : 0) : result;
        }

        private static double sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private static double derivative(double x)
        {
            return x * (1 - x);
        }
    }
}
