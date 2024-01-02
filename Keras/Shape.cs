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
using System.Linq;

namespace Keras
{
    /// <summary>
    /// Keras Tensor Shape
    /// </summary>
    public class Shape
    {
        /// <summary>
        /// Gets the dimensions.
        /// </summary>
        /// <value>
        /// The dimensions.
        /// </value>
        public int[] Dimensions { get; }

        /// <summary>
        /// Initializes a new instance of the <see cref="Shape"/> class.
        /// </summary>
        /// <param name="shape">The shape.</param>
        public Shape(params int[] shape)
        {
            this.Dimensions = shape;
        }

        /// <summary>
        /// Gets the <see cref="System.Int32"/> with the specified n.
        /// </summary>
        /// <value>
        /// The <see cref="System.Int32"/>.
        /// </value>
        /// <param name="n">The n.</param>
        /// <returns></returns>
        public int this[int n] => Dimensions[n];

        #region Equality

        /// <summary>
        /// Implements the operator ==.
        /// </summary>
        /// <param name="a">a.</param>
        /// <param name="b">The b.</param>
        /// <returns>
        /// The result of the operator.
        /// </returns>
        public static bool operator ==(Shape a, Shape b)
        {
            if (b is null) return false;
            return Enumerable.SequenceEqual(a.Dimensions, b?.Dimensions);
        }

        /// <summary>
        /// Implements the operator !=.
        /// </summary>
        /// <param name="a">a.</param>
        /// <param name="b">The b.</param>
        /// <returns>
        /// The result of the operator.
        /// </returns>
        public static bool operator !=(Shape a, Shape b)
        {
            if (b is null) return false;
            return !(a == b);
        }

        /// <summary>
        /// Determines whether the specified <see cref="System.Object" />, is equal to this instance.
        /// </summary>
        /// <param name="obj">The <see cref="System.Object" /> to compare with this instance.</param>
        /// <returns>
        ///   <c>true</c> if the specified <see cref="System.Object" /> is equal to this instance; otherwise, <c>false</c>.
        /// </returns>
        public override bool Equals(object obj)
        {
            if (obj.GetType() != typeof(Shape))
                return false;
            return Enumerable.SequenceEqual(Dimensions, ((Shape)obj).Dimensions);
        }

        /// <summary>
        /// Returns a hash code for this instance.
        /// </summary>
        /// <returns>
        /// A hash code for this instance, suitable for use in hashing algorithms and data structures like a hash table. 
        /// </returns>
        public override int GetHashCode()
        {
            return (Dimensions ?? new int[0]).GetHashCode();
        }

        /// <summary>
        /// Converts to string.
        /// </summary>
        /// <returns>
        /// A <see cref="System.String" /> that represents this instance.
        /// </returns>
        public override string ToString()
        {
            return $"({string.Join(", ", Dimensions ?? new int[0])})";
        }

        /// <summary>
        /// Performs an implicit conversion from <see cref="ValueTuple{System.Int32}"/> to <see cref="Shape"/>.
        /// </summary>
        /// <param name="shape">The shape.</param>
        /// <returns>
        /// The result of the conversion.
        /// </returns>
        public static implicit operator Shape(int shape)
        {
            return new Shape(shape);
        }

        /// <summary>
        /// Performs an implicit conversion from <see cref="ValueTuple{System.Int32, System.Int32}"/> to <see cref="Shape"/>.
        /// </summary>
        /// <param name="shape">The shape.</param>
        /// <returns>
        /// The result of the conversion.
        /// </returns>
        public static implicit operator Shape(ValueTuple<int, int> shape)
        {
            return new Shape(shape.Item1, shape.Item2);
        }

        /// <summary>
        /// Performs an implicit conversion from <see cref="ValueTuple{System.Int32, System.Int32, System.Int32}"/> to <see cref="Shape"/>.
        /// </summary>
        /// <param name="shape">The shape.</param>
        /// <returns>
        /// The result of the conversion.
        /// </returns>
        public static implicit operator Shape(ValueTuple<int, int, int> shape)
        {
            return new Shape(shape.Item1, shape.Item2, shape.Item3);
        }

        /// <summary>
        /// Performs an implicit conversion from <see cref="ValueTuple{System.Int32, System.Int32, System.Int32, System.Int32}"/> to <see cref="Shape"/>.
        /// </summary>
        /// <param name="shape">The shape.</param>
        /// <returns>
        /// The result of the conversion.
        /// </returns>
        public static implicit operator Shape(ValueTuple<int, int, int, int> shape)
        {
            return new Shape(shape.Item1, shape.Item2, shape.Item3, shape.Item4);
        }

        /// <summary>
        /// Performs an implicit conversion from <see cref="ValueTuple{System.Int32, System.Int32, System.Int32, System.Int32, System.Int32}"/> to <see cref="Shape"/>.
        /// </summary>
        /// <param name="shape">The shape.</param>
        /// <returns>
        /// The result of the conversion.
        /// </returns>
        public static implicit operator Shape(ValueTuple<int, int, int, int, int> shape)
        {
            return new Shape(shape.Item1, shape.Item2, shape.Item3, shape.Item4, shape.Item5);
        }

        #endregion
    }
}

