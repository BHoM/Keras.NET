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

using Numpy;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Layers
{
    /// <summary>
    /// This wrapper applies a layer to every temporal slice of an input.
    /// The input should be at least 3D, and the dimension of index one will be considered to be the temporal dimension.
    /// Consider a batch of 32 samples, where each sample is a sequence of 10 vectors of 16 dimensions.
    /// The batch input shape of the layer is then (32, 10, 16), and the input_shape, not including the samples dimension, is (10, 16).
    /// You can then use TimeDistributed to apply a Dense layer to each of the 10 timesteps, independently:
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class TimeDistributed : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TimeDistributed"/> class.
        /// </summary>
        /// <param name="layer">The layer instance.</param>
        public TimeDistributed(BaseLayer layer)
        {
            Parameters["layer"] = layer.PyInstance;

            PyInstance = Keras.keras.layers.TimeDistributed;
            Init();
        }
    }

    /// <summary>
    /// Bidirectional wrapper for RNNs.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class Bidirectional : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Bidirectional" /> class.
        /// </summary>
        /// <param name="layer">The recurrent layer instance.</param>
        /// <param name="merge_mode">Mode by which outputs of the forward and backward RNNs will be combined. One of {'sum', 'mul', 'concat', 'ave', None}. If None, the outputs will not be combined, they will be returned as a list.</param>
        /// <param name="weights">Initial weights to load in the Bidirectional model.</param>
        public Bidirectional(BaseLayer layer, string merge_mode= "concat", NDarray weights= null)
        {
            Parameters["layer"] = layer.PyInstance;

            PyInstance = Keras.keras.layers.Bidirectional;
            Init();
        }
    }
}
