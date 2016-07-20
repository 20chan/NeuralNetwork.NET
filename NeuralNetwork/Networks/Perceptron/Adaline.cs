using System;
using System.Linq;
using NeuralNetwork.Networks.Trains;

namespace NeuralNetwork.Networks.Perceptron
{
    public class Adaline
    {
        private int _layer;
        /// <summary>
        /// 레이어의 뉴럴 개수
        /// </summary>
        public int Layer { get { return _layer; } }
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

        private double[] _weights;
        private double _biasWeight;
        
        public Adaline(int layer)
        {
            _layer = layer;
            _weights = new double[layer];
            Reset();
        }

        /// <summary>
        /// 가중치 초기화
        /// </summary>
        public void Reset()
        {
            Random r = new Random();
            for (int i = 0; i < _layer; i++)
                _weights[i] = r.NextDouble() - 0.5;
            _biasWeight = r.NextDouble() - 0.5;
            _error = double.MaxValue;
        }

        /// <summary>
        /// 학습을 시켜 가중치의 값을 조정합니다.
        /// </summary>
        /// <param name="trainset">학습을 학습 데이터</param>
        /// <param name="learnRate">학습률</param>
        public void Train(BasicTrainSet<double> trainset, double learnRate = 0.03)
        {
            _error = 0;
            _epoch++;
            for (int p = 0; p < trainset.DataCount; p++)
            {
                double result = Compute(trainset.Input[p], false);
                double error = Sigmoid.Derivative(result) * (trainset.Output[p] - result);

                for (int i = 0; i < _weights.Length; i++)
                {
                    _weights[i] += error * trainset.Input[p][i] * learnRate;
                }
                _biasWeight += error * learnRate;
                _error += Math.Abs(error);
            }
        }

        /// <summary>
        /// 입력에 대한 출력값을 계산합니다.
        /// </summary>
        /// <param name="input">입력값</param>
        /// <param name="quan">양자화 여부</param>
        /// <returns>출력값</returns>
        public double Compute(double[] input, bool quan)
        {
            double result = Sigmoid.LogSigmoid(_weights.Zip(input, (a, b) => (a * b)).Sum() + _biasWeight);
            return quan ? (result >= 0.5 ? 1 : 0) : result;
        }

        public void Debug()
        {
            Console.WriteLine($"Epoch : {Epoch} Error : {Error}");
        }
    }
}
