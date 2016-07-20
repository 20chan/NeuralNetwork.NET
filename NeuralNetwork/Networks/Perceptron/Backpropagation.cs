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
            _error = double.MaxValue;
        }

        public void AddLayer(int count)
        {
            Layer layer = new Layer(count);
            if(_layers.Count != 0)
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

                double[] temp = new double[input.Length];
                for (int i = 0; i < input.Length; i++)
                    temp[i] = input[i];

                Layer cur = InputLayer;
                while(cur != null)
                {
                    temp = cur.Compute(temp);
                    cur = cur.NextLayer;
                }

                double result = OutputLayer.Output[0];
                double globalError = Sigmoid.Derivative(result) * (output - result);
                _error += globalError;

                double[][] err = new double[OutputLayer.Count][];
                for (int i = 0; i < OutputLayer.Count; i++)
                    err[i] = new double[] { globalError };
                
                OutputLayer.AdjustWeights(err, learnrate);
                for (int i = 0; i < _layers.Count - 1; i++) //except output layer
                    _layers[i].AdjustWeights(_layers[i + 1].ErrorFeedback(_layers[i]), learnrate);

            }
        }

        public double Compute(double[] input, bool quan)
        {
            Layer cur = InputLayer;

            double[] temp = new double[input.Length];
            for (int i = 0; i < input.Length; i++)
                temp[i] = input[i];

            while (cur != null)
            {
                temp = cur.Compute(temp);
                cur = cur.NextLayer;
            }
            
            return quan ? (OutputLayer.Output[0] >= 0 ? 1: 0) : OutputLayer.Output[0];
        }

        public void DebugWeights()
        {
            for (int i = 0; i < _layers.Count; i++)
            {
                _layers[i].Weights.ToList().ForEach((w) => w.ToList().ForEach((_w) => Console.WriteLine($"{i} 번째 레이어 가중치 {_w}")));
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
        private double[] _bias;

        public double[][] Weights { get { return _weights; } }
        public double[] Bias { get { return _bias; } }

        private double[] _input;
        public double[] Input { get { return _input; } }

        private double[] _output;
        public double[] Output { get { return _output; } }

        public Layer(int count)
        {
            _count = count;
            ResetWeights();
        }

        public void ResetWeights()
        {
            Random r = new Random();

            _weights = new double[Count][];
            _bias = new double[Count];
            for (int i = 0; i < Count; i++)
                _weights[i] = new double[NextLayerCount];

            for (int i = 0; i < Count; i++)
            {
                for (int j = 0; j < NextLayerCount; j++)
                    _weights[i][j] = r.NextDouble() - 0.5;
                _bias[i] = r.NextDouble() - 0.5;
            }
            
        }

        public void LinkTo(Layer next)
        {
            _nextLayer = next;
            ResetWeights();
        }

        public double[] Compute(double[] input)
        {
            _input = input;
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
        
        public void AdjustWeights(double[][] error, double learnRate)
        {
            for (int i = 0; i < Count; i++)
            {
                for (int j = 0; j < NextLayerCount; j++)
                    _weights[i][j] += _weights[i][j] * error[i][j] * Sigmoid.Derivative(Output[j]) * learnRate * Input[j];
                //_bias[i] += error[i] * Sigmoid.Derivative(Output[i]) * learnRate;
            }
        }



        public double[][] ErrorFeedback(Layer layer)
        {
            double[][] result = new double[Count][];
            for (int i = 0; i < Count; i++)
                result[i] = new double[layer.Count];
            for(int i = 0; i < Count; i++)
            {
                for (int j = 0; j < layer.Count; j++)
                {
                    result[i][j] = layer._weights[j][i] * Sigmoid.Derivative(layer.Output[j]);
                }
            }

            return result;
        }

        //Node i -> j, delta weight;
        //delta weight (j)(i) = rl * error * input (i) + [a * delta weight (j)(i)] : momentum term
    }
}
