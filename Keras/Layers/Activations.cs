/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2023, the respective contributors. All rights reserved.
 *
 * Each contributor holds copyright over their respective contributions.
 * The project versioning (Git) records all such contribution source information.
 *                                           
 *                                                                              
 * The BHoM is free software: you can redistribute it and/or modify         
 * it under the terms of the GNU Lesser General Public License as published by  
 * the Free Software Foundation, either version 3.0 of the License, or          
 * (at your option) any later version.                                          
 *                                                                              
 * The BHoM is distributed in the hope that it will be useful,              
 * but WITHOUT ANY WARRANTY; without even the implied warranty of               
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the                 
 * GNU Lesser General Public License for more details.                          
 *                                                                            
 * You should have received a copy of the GNU Lesser General Public License     
 * along with this code. If not, see <https://www.gnu.org/licenses/lgpl-3.0.html>.      
 */

using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Layers
{
    /// <summary>
    /// Applies an activation function to an output.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class Activation : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LeakyReLU"/> class.
        /// </summary>
        /// <param name="alpha">float >= 0. Negative slope coefficient.</param>
        public Activation(PyObject activation)
        {
            Parameters["activation"] = activation;

            PyInstance = Keras.keras.layers.Activation;
            Init();
        }
    }

    /// <summary>
    /// Leaky version of a Rectified Linear Unit.
    /// It allows a small gradient when the unit is not active: f(x) = alpha* x for x< 0, f(x) = x for x >= 0.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class LeakyReLU : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LeakyReLU"/> class.
        /// </summary>
        /// <param name="alpha">float >= 0. Negative slope coefficient.</param>
        public LeakyReLU(float alpha = 0.3f)
        {
            Parameters["alpha"] = alpha;

            PyInstance = Keras.keras.layers.LeakyReLU;
            Init();
        }
    }

    /// <summary>
    /// Parametric Rectified Linear Unit.
    /// It follows: f(x) = alpha* x for x< 0, f(x) = x for x >= 0, where alpha is a learned array with the same shape as x.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class PReLU : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="PReLU" /> class.
        /// </summary>
        /// <param name="alpha_initializer">initializer function for the weights.</param>
        /// <param name="alpha_regularizer">regularizer for the weights.</param>
        /// <param name="alpha_constraint">constraint for the weights.</param>
        /// <param name="shared_axes">the axes along which to share learnable parameters for the activation function. For example, if the incoming feature maps are from a 2D convolution with output shape (batch, height, width, channels), and you wish to share parameters across space so that each filter only has one set of parameters, set shared_axes=[1, 2].</param>
        public PReLU(string alpha_initializer = "zeros", string alpha_regularizer = "", string alpha_constraint = "", int[] shared_axes = null)
        {
            Parameters["alpha_initializer"] = alpha_initializer;
            Parameters["alpha_regularizer"] = alpha_regularizer;
            Parameters["alpha_constraint"] = alpha_constraint;
            Parameters["shared_axes"] = shared_axes;

            PyInstance = Keras.keras.layers.PReLU;
            Init();
        }
    }

    /// <summary>
    /// Exponential Linear Unit.
    /// It follows: f(x) =  alpha* (exp(x) - 1.) for x< 0, f(x) = x for x >= 0.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class ELU : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ELU"/> class.
        /// </summary>
        /// <param name="alpha"> scale for the negative factor.</param>
        public ELU(float alpha = 1)
        {
            Parameters["alpha"] = alpha;

            PyInstance = Keras.keras.layers.LeakyReLU;
            Init();
        }
    }

    /// <summary>
    /// Thresholded Rectified Linear Unit.
    /// It follows: f(x) = x for x > theta, f(x) = 0 otherwise.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class ThresholdedReLU : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ThresholdedReLU" /> class.
        /// </summary>
        /// <param name="theta">float >= 0. Threshold location of activation.</param>
        public ThresholdedReLU(float theta = 1)
        {
            Parameters["theta"] = theta;

            PyInstance = Keras.keras.layers.ThresholdedReLU;
            Init();
        }
    }

    /// <summary>
    /// Softmax activation function.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class Softmax : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Softmax" /> class.
        /// </summary>
        /// <param name="axis"> Integer, axis along which the softmax normalization is applied.</param>
        public Softmax(int axis = -1)
        {
            Parameters["axis"] = axis;

            PyInstance = Keras.keras.layers.Softmax;
            Init();
        }
    }

    /// <summary>
    /// Softmax activation function with logits. Uses the log-sum-exp trick for numerical stability.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class LogSoftmax : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Softmax" /> class.
        /// </summary>
        /// <param name="axis"> Integer, axis along which the softmax normalization is applied.</param>
        public LogSoftmax(int axis = -1)
        {
            Activation activation = new Activation(Keras.tensorflow.nn.log_softmax);

            PyInstance = activation.PyInstance;
        }
    }

    /// <summary>
    /// Sigmoid activation function with logits. Uses the log-sum-exp trick for numerical stability.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class LogSigmoid : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance oyeah f the <see cref="Softmax" /> class.
        /// </summary>
        /// <param name="axis"> Integer, axis along which the softmax normalization is applied.</param>
        public LogSigmoid()
        {
            Activation activation = new Activation(Keras.tensorflow.math.log_sigmoid);

            PyInstance = activation.PyInstance;
        }
    }

    /// <summary>
    /// Rectified Linear Unit activation function.
    /// With default values, it returns element-wise max(x, 0). 
    /// Otherwise, it follows: f(x) = max_value for x >= max_value, f(x) = x for threshold &lt;= x<max_value, f(x) = negative_slope* (x - threshold) otherwise.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class ReLU : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ReLU" /> class.
        /// </summary>
        /// <param name="max_value">float >= 0. Maximum activation value.</param>
        /// <param name="negative_slope">float >= 0. Negative slope coefficient.</param>
        /// <param name="threshold">float. Threshold value for thresholded activation.</param>
        public ReLU(float? max_value = null, float negative_slope = 0, float threshold = 0)
        {
            Parameters["max_value"] = max_value;
            Parameters["negative_slope"] = negative_slope;
            Parameters["threshold"] = threshold;

            PyInstance = Keras.keras.layers.ReLU;
            Init();
        }
    }

    /// <summary>
    /// Sigmoid activation function.
    /// </summary>
    public class Sigmoid : BaseLayer
    {
        /// <summary>
        /// Initialises a new instance of the <see cref="Sigmoid" /> class
        /// </summary>
        public Sigmoid()
        {
            Activation activation = new Activation(Keras.keras.activations.sigmoid);
            PyInstance = activation.PyInstance;
        }
    }

    /// <summary>
    /// Tanh activation function.
    /// </summary>
    public class Tanh : BaseLayer
    {
        /// <summary>
        /// Initialises a new instance of the <see cref="Tanh" /> class
        /// </summary>
        public Tanh()
        {
            Activation activation = new Activation(Keras.keras.activations.tanh);
            PyInstance = activation.PyInstance;
        }
    }
}
