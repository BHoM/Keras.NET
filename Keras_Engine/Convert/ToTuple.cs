/*
 * This file is part of the Buildings and Habitats object Model (BHoM)
 * Copyright (c) 2015 - 2019, the respective contributors. All rights reserved.
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

using System;
using System.Collections.Generic;
using k = Keras;
using BH.oM.DeepLearning;
using BH.oM.DeepLearning.Layers;
using BH.oM.DeepLearning.Activations;
using System.Linq;

namespace BH.Engine.Keras
{
    public static partial class Convert
    {
        /***************************************************/
        /**** Public Methods - Shape                    ****/
        /***************************************************/

        public static object IToTuple(this IShape shape)
        {
            return ToTuple(shape as dynamic);
        }

        /***************************************************/

        public static Tuple<int, int> ToTuple(this Shape2d shape)
        {
            return new Tuple<int, int>(shape.Dim1, shape.Dim2);
        }

        /***************************************************/

        public static Tuple<int, int, int> ToTuple(this Shape3d shape)
        {
            return new Tuple<int, int, int>(shape.Dim1, shape.Dim2, shape.Dim3);
        }

        /***************************************************/
    }
}
