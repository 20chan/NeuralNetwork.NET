using System;
using NeuralNetwork.Networks.Datas;
using NeuralNetwork.Networks.Trains;
using NeuralNetwork.Networks.Perceptron;

namespace NeuralNetwork
{
    static class Program
    {
        static void Main(string[] args)
        {
            TrainAndGate();
            Console.ReadLine();
        }

        private static void TrainAndGate()
        {
            Console.WriteLine("Starting train AND gate...");
            Adaline ada = new Adaline(2);
            
            double[][] inputs = new double[4][]
            {
                new double[] { 0, 0 },
                new double[] { 0, 1 },
                new double[] { 1, 0 },
                new double[] { 1, 1 }
            };
            double[] outputs = new double[4]
            {
                0,
                0,
                0,
                1
            };

            DataSet<double> data = new DataSet<double>(inputs, outputs);
            BasicTrainSet<double> train = new BasicTrainSet<double>(data);

            while (ada.Epoch < 2000 && ada.Error > 0.2)
            {
                ada.Train(train, 0.5);
                ada.Debug();
            }
            Console.WriteLine();
            Console.WriteLine("Training Finished");
            Console.WriteLine($"Epoch : {ada.Epoch}, Error : {ada.Error}");
            Console.WriteLine("0, 0 -> " + ada.Compute(new double[] { 0, 0 }, true));
            Console.WriteLine("0, 1 -> " + ada.Compute(new double[] { 0, 1 }, true));
            Console.WriteLine("1, 0 -> " + ada.Compute(new double[] { 1, 0 }, true));
            Console.WriteLine("1, 1 -> " + ada.Compute(new double[] { 1, 1 }, true));
        }
        
        private static void TrainAndGateBPP()
        {
            Backpropagation bpp = new Backpropagation();
            bpp.AddLayer(2); //input
            bpp.AddLayer(2);
            bpp.AddLayer(2); //output

            Console.WriteLine("Starting train AND gate...");

            double[][] inputs = new double[4][]
            {
                new double[] { 0, 0 },
                new double[] { 0, 1 },
                new double[] { 1, 0 },
                new double[] { 1, 1 }
            };
            double[] outputs = new double[4]
            {
                0,
                0,
                0,
                1
            };

            DataSet<double> data = new DataSet<double>(inputs, outputs);
            BasicTrainSet<double> train = new BasicTrainSet<double>(data);

            while (bpp.Epoch < 2000)
                bpp.Train(train, 0.5);

            Console.WriteLine();
            Console.WriteLine("Training Finished");
            Console.WriteLine($"Epoch : {bpp.Epoch}, Error : {bpp.Error}");
            Console.WriteLine("0, 0 -> " + bpp.Compute(new double[] { 0, 0 }, false));
            Console.WriteLine("0, 1 -> " + bpp.Compute(new double[] { 0, 1 }, false));
            Console.WriteLine("1, 0 -> " + bpp.Compute(new double[] { 1, 0 }, false));
            Console.WriteLine("1, 1 -> " + bpp.Compute(new double[] { 1, 1 }, false));
            bpp.DebugWeights();
            //System.Diagnostics.Debugger.Break();
        }
    }
}
