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

using Numpy;
using Python.Runtime;
using System;
using System.Collections.Generic;
using System.Text;

namespace Keras.Regularizers
{
    /// <summary>
    /// Regularizers allow to apply penalties on layer parameters or layer activity during optimization. These penalties are incorporated in the loss function that the network optimizes.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class L1L2 : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="L1L2"/> class.
        /// </summary>
        /// <param name="l1">The l1.</param>
        /// <param name="l2">The l2.</param>
        public L1L2(float l1 = 0.01f, float l2 = 0.01f)
        {
            Parameters["l1"] = l1;
            Parameters["l2"] = l2;
            PyInstance = Keras.keras.regularizers.L1L2;
            Init();
        }
    }

    /// <summary>
    /// Regularizers allow to apply penalties on layer parameters or layer activity during optimization. These penalties are incorporated in the loss function that the network optimizes.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class L1 : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="L1"/> class.
        /// </summary>
        /// <param name="l1">The l1.</param>
        public L1(float l1 = 0.01f)
        {
            Parameters["l1"] = l1;
            PyInstance = Keras.keras.regularizers.L1;
            Init();
        }
    }

    /// <summary>
    /// Regularizers allow to apply penalties on layer parameters or layer activity during optimization. These penalties are incorporated in the loss function that the network optimizes.
    /// </summary>
    /// <seealso cref="Keras.Base" />
    public class L2 : Base
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="L2" /> class.
        /// </summary>
        /// <param name="l2">The l2.</param>
        public L2(float l2 = 0.01f)
        {
            Parameters["l2"] = l2;
            PyInstance = Keras.keras.regularizers.L2;
            Init();
        }
    }
}

