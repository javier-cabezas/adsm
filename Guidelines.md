# Introduction #

This document discusses the coding style and some design decisions done in GMAC


# Locking #

All locking within GMAC is done through the `gmac::util::Lock` and `gmac::util::RWLock` classes. These two classes are defined in `src/util/Lock.h`. Any class requiring locking MUST inherit from any of these two classes.
```
class SimpleLockingClass : public gmac::util::Lock {
 . . .
};

class ComplexLockingClass : public gmac::util::RWLock {
 . . .
};
```
Moreover, each lock MUST have its own Paraver name, which is assigned in the constructor. Paraver names are defined in `src/config/paraver.h` and `src/gmac/paraver.c`. When a new class with locks is added, the corresponding Paraver names MUST be added to these two files. Following the previous example, the constructors for the previous classes would look like:
```
SimpleLockingClass::SimpleLockingClass() : gmac::util::Lock(SimpleLockName) {};
ComplexLockingClass::ComplexLockingClass() : gmac::util::Lock(ComplexLockName) {};
```

Sometimes, an object requires locking some of its own attributes. For instance, consider the following class:
```
class MyClass {
   SimpleLockingClass __attribute;
   . . .
   void foo();
};

void MyClass::foo() {
   . . .
   __attribute.lock();
   . . .
};
```
This special case is handled by modifying the declaration of `SimpleLockingClass` to define `MyClass` as a **friend class**. This friendship is required to have a simple way of knowing which external classes are able to lock each class in GMAC. This constrain in GMAC intends to ease the debugging of dead-locks.
```
class SimpleLockingClass : public gmac::util::Lock {
protected:
 friend class MyClass;
 . . .
};
```

# Logging #

Logging in GMAC is done using the `gmac::util::Logger` class, which is declared in `src/util/Logger.h`. Any class that requires dumping debugging information MUST inherit from `gmac::util::Logger`.
```
class MyClass : public gmac::util::Logger {
 . . .
};
```

The `gmac::util::Logger` class implements five main methods:
  * **trace**: prints debugging information that is only visible when GMAC is compiled with the `--enable-debug` flag.
  * **warning**: prints warnings when GMAC is compiled with the `--enable-debug` flag.
  * **assertion**: aborts the program execution if the a condition is false when GMAC is compiled with the `--enable-debug` flag.
  * **fatal**: notifies a fatal error, even when GMAC is compiled without the `--enable-debug` flags.
  * **cfatal**: notifies a fatal error if a conditional is not met, even when GMAC is compiled without the `--enable-debug` flag.

GMAC also implements a global logger, `gmac::util::Logger`, that MUST be only used from static and global functions. This global logger implements the same methods than local loggers, but they must be called capitalized (i.e., TRACE, WARNING, ASSERTION, FATAL, CFATAL).

# STL #
Quite often STL structures are used by GMAC. Unfortunately, STL is not thread-safe, so STL classes might become a huge source of race conditions, unless locking mechanisms are added. STL classes are incorporated in GMAC using private inheritance, and wrapper methods are implemented to provide thread-safe access to these data structures.

```
class MyQueue : private std::list<MyType>, public gmac::util::RWLock {
private:
 typedef std::list<MyType> Parent;
 . . .
public:
 inline bool insert(const MyType &t) {
    lockWrite();
    bool ret = Parent::insert(t);
    unlock();
    return ret;
 }
```