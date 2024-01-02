/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2024, the respective contributors. All rights reserved.
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
        public Sequential(BaseLayer[] layers) : this()
        {
            foreach (var item in layers)
            {
                PyInstance.add(layer: item.PyInstance);
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

