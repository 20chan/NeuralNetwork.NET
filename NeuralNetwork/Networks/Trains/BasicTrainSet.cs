using System;
using NeuralNetwork.Networks.Datas;

namespace NeuralNetwork.Networks.Trains
{
    public class BasicTrainSet<T> : TrainSet
    {
        private DataSet<T> _trainDataSet;
        //public DataSet<T> TrainDataSet { get { return _trainDataSet; } set { _trainDataSet = value; } }

        public int DataCount { get { return _trainDataSet.Count; } }

        public T[][] Input { get { return _trainDataSet.Input; } }
        public T[] Output { get { return _trainDataSet.Output; } }

        public BasicTrainSet(DataSet<T> dataset) : base()
        {
            _trainDataSet = dataset;
        }
    }
}