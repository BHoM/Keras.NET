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
using System.Linq;

namespace Keras
{
    public abstract class Base
    {
        internal dynamic PyInstance;
        public Dictionary<string, object> Parameters = new Dictionary<string, object>();

        public object None = null;

        public void Init()
        {
            PyInstance = Instantiate();
        }

        public virtual PyObject Instantiate()
        {
            var pyargs = Keras.ToTuple(new object[]
            {
                Parameters.FirstOrDefault().Value
            });

            var kwargs = new PyDict();

            bool skip = true;
            foreach (var item in Parameters)
            {
                if (skip)
                {
                    skip = false;
                    continue;
                }

                if (item.Value != null && !string.IsNullOrWhiteSpace(item.Value.ToString()))
                {
                    kwargs[item.Key] = Keras.ToPython(item.Value);
                }
            }

            if (Parameters.Count > 0)
                return PyInstance.Invoke(pyargs, kwargs);
            else
                return PyInstance.Invoke(null, null);
        }

        public virtual PyObject ToPython()
        {
            return PyInstance;
        }

        public static PyObject InvokeStaticMethod(dynamic caller, string method, Dictionary<string, object> args)
        {
            var pyargs = Keras.ToTuple(new object[]
            {
                args.FirstOrDefault().Value
            });

            var kwargs = new PyDict();

            bool skip = true;
            foreach (var item in args)
            {
                if (skip)
                {
                    skip = false;
                    continue;
                }

                if (item.Value != null && !string.IsNullOrWhiteSpace(item.Value.ToString()))
                {
                    kwargs[item.Key] = Keras.ToPython(item.Value);
                }
            }

            if (args.Count > 0)
                return caller.InvokeMethod(method, pyargs, kwargs);
            else
                return caller.InvokeMethod(method, null, null);
        }

        public PyObject InvokeMethod(string method, Dictionary<string, object> args)
        {
           var pyargs = Keras.ToTuple(new object[]
           {
                args.FirstOrDefault().Value
           });

            var kwargs = new PyDict();

            bool skip = true;
            foreach (var item in args)
            {
                if (skip)
                {
                    skip = false;
                    continue;
                }

                if (item.Value != null && !string.IsNullOrWhiteSpace(item.Value.ToString()))
                {
                    kwargs[item.Key] = Keras.ToPython(item.Value);
                }
            }

            if (args.Count > 0)
                return PyInstance.InvokeMethod(method, pyargs, kwargs);
            else
                return PyInstance.InvokeMethod(method, null, null);
        }

        public object this[string name]
        {
            get
            {
                return Parameters[name];
            }
            set
            {
                Parameters[name] = value;
            }
        }
    }
}
