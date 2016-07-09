using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetwork.Networks.Datas
{
    public class DataSet<T>
    {
        public readonly T[][] Input;
        public readonly T[] Output;

        public int Count { get { return Input.Length; } }

        public DataSet(T[][] input, T[] output)
        {
            this.Input = input;
            this.Output = output;
        }
    }
}
