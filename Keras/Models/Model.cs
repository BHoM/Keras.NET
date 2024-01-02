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

namespace Keras.Models
{
    using global::Keras.Layers;
    using Python.Runtime;
    using System.Collections.Generic;

    /// <summary>
    /// In the functional API, given some input tensor(s) and output tensor(s).
    /// This model will include all layers required in the computation of b given a.
    /// </summary>
    /// <seealso cref="Keras.Models.BaseModel" />
    public class Model : BaseModel
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="Model"/> class.
        /// </summary>
        internal Model()
        {

        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Model" /> class.
        /// </summary>
        /// <param name="obj">The object.</param>
        internal Model(PyObject obj)
        {
            PyInstance = obj;
            Init();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Model"/> class.
        /// </summary>
        /// <param name="inputs">The inputs layers.</param>
        /// <param name="outputs">The outputs layers.</param>
        public Model(BaseLayer[] inputs, BaseLayer[] outputs)
        {
            List<PyObject> inputList = new List<PyObject>();
            List<PyObject> outputList = new List<PyObject>();

            foreach (var item in inputs)
            {
                inputList.Add(item.PyInstance);
            }

            foreach (var item in outputs)
            {
                outputList.Add(item.PyInstance);
            }

            PyInstance = Keras.keras.models.Model(inputs[0], outputs);
            Init();
        }
    }
}

