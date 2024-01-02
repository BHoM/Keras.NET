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

using System;
using k = Keras;
using System.Collections.Generic;
using BH.oM.DeepLearning;
using Numpy;
using System.Linq;

namespace BH.Engine.Keras
{
    public static partial class Query
    {
        /***************************************************/
        /**** Public Methods                            ****/
        /***************************************************/

        public static NDarray LoadImage(string path, Shape2d resize = null, ImageFormat format = ImageFormat.ChannelFirst,
                                        InterpolationMethod interpolation = InterpolationMethod.Nearest, bool addbatchDimension = false)
        {
            // using dyamic for now since PIL has not been ported to C#.
            // In the future we want to create a PIL.Image class and return that
            dynamic pilImage = k.PreProcessing.Image.ImageUtil.LoadImg(path, color_mode: "rgb", target_size: resize?.ToKeras(), interpolation: interpolation.ToKeras());
            NDarray array = k.PreProcessing.Image.ImageUtil.ImageToArray(pilImage, format.ToKeras());
            if (addbatchDimension)
                array = np.expand_dims(array, 0);
            return array;
        }

        /***************************************************/

        public static NDarray LoadImageBatch(List<string> paths, Shape2d imageSize, ImageFormat format = ImageFormat.ChannelFirst,
                                             InterpolationMethod interpolation = InterpolationMethod.Nearest)
        {
            NDarray array;
            switch (format)
            {
                case ImageFormat.ChannelFirst:
                    array = np.empty(paths.Count(), 3, imageSize.Dim1, imageSize.Dim2);
                    break;
                case ImageFormat.ChannelLast:
                    array = np.empty(paths.Count(), imageSize.Dim1, imageSize.Dim2, 3);
                    break;
                case ImageFormat.GreyScale:
                    array = np.empty(paths.Count(), imageSize.Dim1, imageSize.Dim2);
                    break;
                default:
                    Engine.Reflection.Compute.RecordError($"Invalid format {format} specified");
                    return null;
            }
            
            for (int i = 0; i < paths.Count; i++)
                array[i] = LoadImage(paths[i], imageSize, format, interpolation);

            return array;
        }

        /***************************************************/

    }
}


