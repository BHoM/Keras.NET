using Keras.Layers;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Models
{
    /// <summary>
    /// The Sequential model is a linear stack of layers. You can create a Sequential model by passing a list of layer instances to the constructor
    /// </summary>
    /// <seealso cref="Keras.Models.BaseModel" />
    public class Sequential : BaseModel
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Sequential"/> class.
        /// </summary>
        internal Sequential(PyObject obj)
        {
            PyInstance = obj;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Sequential"/> class.
        /// </summary>
        public Sequential()
        {
            PyInstance = Keras.keras.models.Sequential();
            //Init();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Sequential"/> class.
        /// </summary>
        /// <param name="layers">The layers.</param>
        public Sequential(Base[] layers) : this()
        {
            foreach (var item in layers)
            {
                Add(item);
            }
        }

        /// <summary>
        /// Stacks a layer or loss operation on the top of the current ones
        /// </summary>
        /// <param name="layer">The layer.</param>
        public void Add(Base module)
        {
            switch (module)
            {
                case BaseLayer layer:
                    Add(layer); return;
                case BaseLoss loss:
                    Add(loss); return;
                default:
                    throw new NotImplementedException($"module must be an instance of k.Layers.BaseLayer or k.Layers.BaseLoss, not {module.GetType()}");
            }
        }

        /// <summary>
        /// You can also simply add layers via the .Add() method
        /// </summary>
        /// <param name="layer">The layer.</param>
        public void Add(BaseLayer layer)
        {
            PyInstance.add(layer: layer.PyInstance);
        }

        /// <summary>
        /// You can also losses via the .Add() method
        /// </summary>
        /// <param name="loss">The loss function.</param>
        public void Add(BaseLoss loss)
        {
            PyInstance.add_loss(losses: loss.PyInstance);
        }
    }
}
