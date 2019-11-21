using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Python.Runtime;

namespace Keras.Layers
{
    public class Merge: BaseLayer
    {
        
    }

    public class Add : Merge
    {
        public Add(params BaseLayer[] inputs)
        {
            //Parameters["inputs"] = inputs;
            PyInstance = Keras.keras.layers.add(inputs: inputs.Select(x=>(x.PyInstance)).ToArray());
        }
    }

    public class Concatenate : Merge
    {
        public Concatenate(params BaseLayer[] inputs)
        {
            //Parameters["inputs"] = inputs;
            PyInstance = Keras.keras.layers.concatenate(inputs.Select(x => (x.PyInstance)).ToArray());
        }
    }
}
