/**
 * @brief A solver factory that allows one to register solvers, similar to
 * layer factory. During runtime, registered solvers could be called by passing
 * a SolverParameter protobuffer to the CreateSolver function:
 *
 *     SolverRegistry<Dtype>::CreateSolver(param);
 *
 * There are two ways to register a solver. Assuming that we have a solver like:
 *
 *   template <typename Dtype>
 *   class MyAwesomeSolver : public Solver<Dtype> {
 *     // your implementations
 *   };
 *
 * and its type is its C++ class name, but without the "Solver" at the end
 * ("MyAwesomeSolver" -> "MyAwesome").
 *
 * If the solver is going to be created simply by its constructor, in your c++
 * file, add the following line:
 *
 *    REGISTER_SOLVER_CLASS(MyAwesome);
 *
 * Or, if the solver is going to be created by another creator function, in the
 * format of:
 *
 *    template <typename Dtype>
 *    Solver<Dtype*> GetMyAwesomeSolver(const SolverParameter& param) {
 *      // your implementation
 *    }
 *
 * then you can register the creator function instead, like
 *
 * REGISTER_SOLVER_CREATOR(MyAwesome, GetMyAwesomeSolver)
 *
 * Note that each solver type should only be registered once.
 */

#ifndef CAFFE_SOLVER_FACTORY_H_
#define CAFFE_SOLVER_FACTORY_H_

#include <map>
#include <string>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

template <typename Dtype>
class Solver;

template <typename Dtype>
class SolverRegistry {
 public:
  //Creator是一个函数指针类型，指向的函数的参数为SolverParameter类型
  //，返回类型为Solver<Dtype>*
  typedef Solver<Dtype>* (*Creator)(const SolverParameter&);

  //
  typedef std::map<string, Creator> CreatorRegistry;

  static CreatorRegistry& Registry() {
	//静态变量
    static CreatorRegistry* g_registry_ = new CreatorRegistry();
    return *g_registry_;
  }

  // Adds a creator.
  // 添加一个creator
  static void AddCreator(const string& type, Creator creator) {
    CreatorRegistry& registry = Registry();
    CHECK_EQ(registry.count(type), 0)
        << "Solver type " << type << " already registered.";
    registry[type] = creator;
  }

  // Get a solver using a SolverParameter.
  // 静态成员函数，在caffe.cpp里直接调用，返回Solver指针
  static Solver<Dtype>* CreateSolver(const SolverParameter& param) {
	// string类型的变量type，表示Solver的类型(‘SGD’/’Nestrov’等)
	// 默认为SGD
    const string& type = param.type();
    // 定义了一个key类型为string，value类型为Creator的map：registry
    // 返回为静态变量
    CreatorRegistry& registry = Registry();


    for (typename CreatorRegistry::iterator iter = registry.begin();
             iter != registry.end(); ++iter)
    {
         std::cout<<"key:"<<iter->first<<"``` "
             <<"value:"<<iter->second<<std::endl;}

    /*
     * 如果是一个已经register过的Solver类型，那么registry.count(type)应该为1，
     * 然后通过registry这个map返回了我们需要类型的Solver的creator，并调用这个
     * creator函数，将creator返回的Solver<Dtype>*返回。
     */
    CHECK_EQ(registry.count(type), 1) << "Unknown solver type: " << type
        << " (known types: " << SolverTypeListString() << ")";
    //通过static的g_registry_[type]获得type对应的solver的creator函数指针
    return registry[type](param);
  }

  static vector<string> SolverTypeList() {
    CreatorRegistry& registry = Registry();
    vector<string> solver_types;
    for (typename CreatorRegistry::iterator iter = registry.begin();
         iter != registry.end(); ++iter) {
      solver_types.push_back(iter->first);
    }
    return solver_types;
  }

 private:
  // Solver registry should never be instantiated - everything is done with its
  // static variables.
  // Solver registry不应该被实例化，因为所有的成员都是静态变量
  // 构造函数是私有的，所有成员函数都是静态的，可以通过类调用
  SolverRegistry() {}

  static string SolverTypeListString() {
    vector<string> solver_types = SolverTypeList();
    string solver_types_str;
    for (vector<string>::iterator iter = solver_types.begin();
         iter != solver_types.end(); ++iter) {
      if (iter != solver_types.begin()) {
        solver_types_str += ", ";
      }
      solver_types_str += *iter;
    }
    return solver_types_str;
  }
};


template <typename Dtype>
class SolverRegisterer {
 public:
  SolverRegisterer(const string& type,
		  	  	  	 // 指针函数
                     Solver<Dtype>* (*creator)(const SolverParameter&))
  {
    // LOG(INFO) << "Registering solver type: " << type;
    SolverRegistry<Dtype>::AddCreator(type, creator);
  }
};

/*
分别定义了SolverRegisterer这个模板类的float和double类型的static变量，这会去调用各自
的构造函数，而在SolverRegisterer的构造函数中调用了之前提到的SolverRegistry类的
AddCreator函数，这个函数就是将刚才定义的Creator_SGDSolver这个函数的指针存到
g_registry指向的map里面。
*/
#define REGISTER_SOLVER_CREATOR(type, creator)                                 \
  static SolverRegisterer<float> g_creator_f_##type(#type, creator<float>);    \
  static SolverRegisterer<double> g_creator_d_##type(#type, creator<double>)   \
/*
这个宏会定义一个名为Creator_SGDSolver的函数，这个函数即为Creator类型的指针指向的函数，
在这个函数中调用了SGDSolver的构造函数，并将构造的这个变量得到的指针返回，这也就是Creator
类型函数的作用：构造一个对应类型的Solver对象，将其指针返回。然后在这个宏里又调用了
REGISTER_SOLVER_CREATOR这个宏
*/
#define REGISTER_SOLVER_CLASS(type)                                            \
  template <typename Dtype>                                                    \
  Solver<Dtype>* Creator_##type##Solver(                                       \
      const SolverParameter& param)                                            \
  {                                                                            \
    return new type##Solver<Dtype>(param);                                     \
  }                                                                            \
  REGISTER_SOLVER_CREATOR(type, Creator_##type##Solver)

}  // namespace caffe

#endif  // CAFFE_SOLVER_FACTORY_H_
