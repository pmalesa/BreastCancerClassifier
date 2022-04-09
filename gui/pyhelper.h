#ifndef PYHELPER_H
#define PYHELPER_H
#pragma once

#undef slots
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define slots Q_SLOTS

class CPyInstance
{
public:
    CPyInstance()
    {
        Py_Initialize();
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append(\".\")");
    }
    ~CPyInstance()
    {
        Py_Finalize();
    }
};

class CPyObject
{
private:
    PyObject* p_;

public:
    CPyObject() : p_(nullptr)   { }
    CPyObject(PyObject* p) : p_(p)  { }
    ~CPyObject()
    {
        Release();
    }

    PyObject* getObject()
    {
        return p_;
    }

    PyObject* setObject(PyObject* p)
    {
        return (p_ = p);
    }

    PyObject* AddRef()
    {
        if (p_)
        {
            Py_INCREF(p_);
        }
        return p_;
    }

    void Release()
    {
        if (p_)
        {
            Py_DECREF(p_);
        }
        p_ = nullptr;
    }

    PyObject* operator ->()
    {
        return p_;
    }

    bool is()
    {
        return p_ ? true : false;
    }

    operator PyObject*()
    {
        return p_;
    }

    PyObject* operator = (PyObject* p)
    {
        p_ = p;
        return p_;
    }

    operator bool()
    {
        return p_ ? true : false;
    }
};


#endif // PYHELPER_H
