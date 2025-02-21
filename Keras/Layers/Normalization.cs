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
using System.Collections.Generic;
using System.Text;

namespace Keras.Layers
{
    /// <summary>
    /// Batch normalization layer (Ioffe and Szegedy, 2014).
    /// Normalize the activations of the previous layer at each batch, i.e.applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1.
    /// </summary>
    /// <seealso cref="Keras.Layers.BaseLayer" />
    public class BatchNormalization : BaseLayer
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="BatchNormalization"/> class.
        /// </summary>
        /// <param name="axis"> Integer, the axis that should be normalized (typically the features axis). For instance, after a Conv2D layer with data_format="channels_first", set axis=1 in BatchNormalization.</param>
        /// <param name="momentum"> Momentum for the moving mean and the moving variance.</param>
        /// <param name="epsilon"> Small float added to variance to avoid dividing by zero.</param>
        /// <param name="center"> If True, add offset of beta to normalized tensor. If False, beta is ignored.</param>
        /// <param name="scale"> If True, multiply by gamma. If False, gamma is not used. When the next layer is linear (also e.g. nn.relu), this can be disabled since the scaling will be done by the next layer.</param>
        /// <param name="beta_initializer"> Initializer for the beta weight.</param>
        /// <param name="gamma_initializer"> Initializer for the gamma weight.</param>
        /// <param name="moving_mean_initializer"> Initializer for the moving mean.</param>
        /// <param name="moving_variance_initializer"> Initializer for the moving variance.</param>
        /// <param name="beta_regularizer"> Optional regularizer for the beta weight.</param>
        /// <param name="gamma_regularizer"> Optional regularizer for the gamma weight.</param>
        /// <param name="beta_constraint"> Optional constraint for the beta weight.</param>
        /// <param name="gamma_constraint"> Optional constraint for the gamma weight</param>
        public BatchNormalization(int axis= -1, float momentum= 0.99f, float epsilon= 0.001f, bool center = true, bool scale= true, string beta_initializer= "zeros",
                                string gamma_initializer= "ones", string moving_mean_initializer= "zeros", string moving_variance_initializer= "ones",
                                string beta_regularizer= null, string gamma_regularizer= null, string beta_constraint= null, string gamma_constraint= null, Shape input_shape = null)

        {
            if (input_shape != null)
            {
                PyInstance = Keras.keras.layers.BatchNormalization(axis: axis, momentum: momentum, epsilon: epsilon, center: center,
                                                        scale: scale, beta_initializer: beta_initializer, gamma_initializer: gamma_initializer,
                                                        moving_mean_initializer: moving_mean_initializer, moving_variance_initializer: moving_variance_initializer,
                                                        beta_regularizer: beta_regularizer, gamma_regularizer: gamma_regularizer, beta_constraint: beta_constraint,
                                                        gamma_constraint: gamma_constraint, input_shape: input_shape);
            }
            else
            {
                PyInstance = Keras.keras.layers.BatchNormalization(axis: axis, momentum: momentum, epsilon: epsilon, center: center,
                                                       scale: scale, beta_initializer: beta_initializer, gamma_initializer: gamma_initializer,
                                                       moving_mean_initializer: moving_mean_initializer, moving_variance_initializer: moving_variance_initializer,
                                                       beta_regularizer: beta_regularizer, gamma_regularizer: gamma_regularizer, beta_constraint: beta_constraint,
                                                       gamma_constraint: gamma_constraint);
            }
        }
    }
}

