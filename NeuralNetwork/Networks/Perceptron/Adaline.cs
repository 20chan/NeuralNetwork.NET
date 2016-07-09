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

        public void Train(BasicTrainSet<double> trainset, double learnRate = 0.04)
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

    public class Layer
    {
        double[] _weights; //0번째는 바이어스 가중치
        int _count;
        double _bias;
        Func<double, double> _sigmoid;
        double _error;

        public double[] Weights { get { return _weights; } }
        public int Count { get { return _count; } }
        public Layer NextLayer { get; set; }

        public Layer(int n, int next, Func<double, double> sigmoid, double bias = -1)
        {
            if (n < 0) throw new Exception("뉴럴의 개수는 0개 이상이어야 합니다.");
            _count = n + 1;
            _weights = new double[n + 1]; //바이어스
            _bias = bias;

            _sigmoid = sigmoid;
            NextLayer = null;

            Init();
        }

        public void Init()
        {
            Random r = new Random();
            for (int i = 0; i < _count; i++)
                    _weights[i] = r.NextDouble();
        }
        
        public void Train(int epoch, double learnRate, double[][] inputs, double[] outputs)
        {
            for (int p = 0; p < epoch; p++)
            {
                for (int i = 0; i < inputs.GetLength(0); i++)
                {
                    _error = 0;
                    double res = Eval(inputs[i]);
                    if (res == outputs[i])
                        continue;
                    else
                    {
                        double error = outputs[i] - res;
                        _weights[0] = _weights[0] + learnRate * _bias * error;
                        for (int k = 1; k < _count; k++)
                            _weights[k] = _weights[k] + learnRate * inputs[i][k - 1] * error;
                        _error += eval(inputs[i]);
                    }

                }
            }
        }
        /*
        public void Train(int epoch, double learnRate, double[][] inputs, double[] outputs)
        {
            for (int p = 0; p < epoch; p++)
            {
                for (int i = 0; i < inputs.GetLength(0); i++)
                {
                    double[] firstRes = evals(inputs[i]);
                    double[] _res = evals(inputs[i]);

                    double[] _r = new double[_res.GetLength(0)];
                    for (int k = 0; k < _res.GetLength(0); k++)
                        _r[k] = _res[k];
                    _res = evals(_r);

                    double res = 0;
                    for (int k = 0; k < _count; k++)
                        res += _res[k];
                    res = res >= 0 ? 1 : 0;

                    if (res == outputs[i])
                        continue;
                    else
                    {
                        double error = outputs[i] - res;
                        _weights[0] = _weights[0] + learnRate * _bias * error;
                        for (int k = 1; k < _count; k++)
                            _weights[k] = _weights[k] + learnRate * inputs[i][k - 1] * error;
                        _error += eval(inputs[i]);
                        
                    }

                }
            }
        }
        */
        public double Eval(double[] input)
        {
            double res = 0;
            double[] result = evals(input);
            
                double[] _r = new double[result.GetLength(0)];
                for (int i = 0; i < result.GetLength(0); i++)
                    _r[i] = result.Sum();
                result = evals(_r);

            for (int i = 0; i < _count; i++)
                res += result[i];

            return res;
        }

        private double eval(double[] input)
        {
            double res = 0;
            double[] result = evals(input);
            for (int i = 0; i < _count; i++)
                res += result[i];
            return res;
        }

        private double[] evals(double[] input)
        {
            double[] result = new double[_count];
            
            result[0] = Math.Tanh(_weights[0] * _bias);
            for (int j = 0; j < _count - 1; j++)
                result[j + 1] = _sigmoid(input[j] * _weights[j + 1]);

            return result;
        }
    }
}
