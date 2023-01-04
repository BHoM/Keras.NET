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

using BH.Engine.Reflection;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using k = Keras;

namespace BH.Engine.Keras
{
    public static partial class External
    {
        /***************************************************/
        /**** Public Methods                            ****/
        /***************************************************/

        public static List<MethodInfo> Methods()
        {
            Dictionary<string, MethodInfo> methodsDictionary = new Dictionary<string, MethodInfo>();

            Type[] typesToExplore = typeof(k.Keras).Assembly.GetTypes();

            foreach (Type type in typesToExplore)
            {
                List<MethodInfo> typeMethods = type.NestedMethods();
                for (int i = 0; i < typeMethods.Count; i++)
                {
                    MethodInfo methodInstance = typeMethods[i];
                    if (!methodsDictionary.ContainsKey(methodInstance.Name) && !methodInstance.DeclaringType.Namespace.Contains("System"))
                        methodsDictionary.Add(methodInstance.Name, methodInstance);
                }
            }

            List<MethodInfo> methods = methodsDictionary.Values.ToList();

            return methods.Where(x => x != null).ToList();
        }


        /***************************************************/
        /**** Private Fields                            ****/
        /***************************************************/

        public static List<Type> typesToExplore = new List<Type>
        {
            typeof(k.Activations),
            typeof(k.Base),
            typeof(k.DirectoryIterator),
            typeof(k.Keras),
            typeof(k.KerasIterator),
            typeof(k.Losses),
            typeof(k.Metrics),
            //typeof(k.NumpyExtension),
            typeof(k.Shape),
            typeof(k.Applications.DenseNet.DenseNet121),
            typeof(k.Applications.DenseNet.DenseNet169),
            typeof(k.Applications.DenseNet.DenseNet201),
            typeof(k.Applications.Inception.InceptionResNetV2),
            typeof(k.Applications.Inception.InceptionV3),
            typeof(k.Applications.MobileNet.MobileNetV1),
            typeof(k.Applications.MobileNet.MobileNetV2),
            typeof(k.Applications.NASNet.NASNetLarge),
            typeof(k.Applications.NASNet.NASNetMobile),
            typeof(k.Applications.ResNet.ResNet101),
            typeof(k.Applications.ResNet.ResNet152),
            typeof(k.Applications.ResNet.ResNet50),
            typeof(k.Applications.ResNetV2.ResNet101V2),
            typeof(k.Applications.ResNetV2.ResNet152V2),
            typeof(k.Applications.ResNetV2.ResNet50V2),
            typeof(k.Applications.ResNext.ResNeXt101),
            typeof(k.Applications.ResNext.ResNeXt50),
            typeof(k.Applications.VGG.VGG16),
            typeof(k.Applications.VGG.VGG19),
            typeof(k.Applications.Xception),
            typeof(k.Applications.ImageNetPrediction),
            typeof(k.Datasets.BostonHousing),
            typeof(k.Datasets.Cifar10),
            typeof(k.Datasets.Cifar100),
            typeof(k.Datasets.FashionMNIST),
            typeof(k.Datasets.IMDB),
            typeof(k.Datasets.MNIST),
            typeof(k.Datasets.Reuters),
        };

        /***************************************************/
    }
}

