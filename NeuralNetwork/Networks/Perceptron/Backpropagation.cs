using System;
using System.Collections.Generic;
using System.Linq;
using NeuralNetwork.Networks.Trains;

namespace NeuralNetwork.Networks.Perceptron
{
    public class Backpropagation
    {
        private List<Layer> _layers;

        public Layer InputLayer { get { return _layers.First(); } }
        public Layer OutputLayer { get { return _layers.Last(); } }

        private int _epoch;
        /// <summary>
        /// 학습시킨 수
        /// </summary>
        public int Epoch { get { return _epoch; } }

        private double _error;
        /// <summary>
        /// 마지막 학습시켰을 때 계산된 오차율
        /// </summary>
        public double Error { get { return _error; } }

        public Backpropagation()
        {
            Reset();
        }

        public void Reset()
        {
            _layers = new List<Layer>();
        }

        public void AddLayer(Layer layer)
        {
            OutputLayer.LinkTo(layer);
            _layers.Add(layer);
        }

        public void Train(BasicTrainSet<double> trainset, double learnrate = 0.03)
        {
            _epoch++;
            _error = 0;
            for (int p = 0; p < trainset.DataCount; p++)
            {
                double[] input = trainset.Input[p];
                double output = trainset.Output[p];

                double[] temp = new double[trainset.DataCount];
                for (int i = 0; i < trainset.DataCount; i++)
                    temp[i] = input[i];

                Layer cur = InputLayer;
                while(cur != OutputLayer)
                {
                    temp = cur.Compute(temp);
                    cur = cur.NextLayer;
                }

                double result = OutputLayer.Output[0];
                double outputError = Sigmoid.Derivative(result) * (output - result);
            }
        }
    }

    public class Layer
    {
        private int _count;
        public int Count { get { return _count; } }

        private Layer _nextLayer;
        public Layer NextLayer { get { return _nextLayer; } }
        public int NextLayerCount { get { return NextLayer == null ? 1 : NextLayer.Count; } }

        private double[][] _weights;
        private double _bias;

        private double[] _output;
        public double[] Output { get { return _output; } }

        private double _error;
        public double Error { get { return _error; } }

        public Layer(int count)
        {
            _count = count;
        }

        public void ResetWeights()
        {
            Random r = new Random();

            _weights = new double[Count][];
            for (int i = 0; i < Count; i++)
                _weights[i] = new double[NextLayerCount];

            for (int i = 0; i < Count; i++)
                for (int j = 0; j < Count; j++)
                    _weights[i][j] = r.NextDouble() - 0.5;

            _bias = r.NextDouble() - 0.5;
        }

        public void LinkTo(Layer next)
        {
            _nextLayer = next;
            ResetWeights();
        }

        public double[] Compute(double[] input)
        {
            double[] result = new double[NextLayerCount];
            _output = new double[NextLayerCount];

            for (int i = 0; i < NextLayerCount; i++)
            {
                for(int j = 0; j < Count; j++)
                {
                    result[i] += _weights[j][i] * input[j];
                }
                result[i] = Sigmoid.LogSigmoid(result[i]);
                _output[i] = result[i];
            }

            return result;
        }

        public double GetError(double outlayerError)
        {
            throw new NotImplementedException();
        }

        //Node i -> j, delta weight;
        //delta weight (j)(i) = rl * error * input (i) + [a * delta weight (j)(i)] : momentum term
    }
}
