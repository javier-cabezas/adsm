#ifndef TEST_H
#define TEST_H

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include <sys/types.h>
#include <sys/wait.h>

#include "Variable.h"

class Test
{
private:
    class TestCase {
        typedef std::pair<std::string, std::string> KeyVal;
    private:
        std::vector<KeyVal> keyvals_;
        std::string name_;
        int status_;
        bool run_;

        void setEnvironment()
        {
            std::vector<KeyVal>::const_iterator it;
            for (it = keyvals_.begin(); it != keyvals_.end(); it++) {
                ::setenv(it->first.c_str(), it->second.c_str(), 1);
            }
        }

    public:
        TestCase() :
            name_(""),
            run_(false)
        {}

        void addVariableInstance(std::string name, std::string value)
        {
            keyvals_.push_back(KeyVal(name, value));
            name_ += name + "=" + value + ";";
        }

        bool isFailure() const
        {
            return status_ != 0;
        }

        std::string getName() const
        {
            return name_;
        }

        void run(std::string exec);

        void report(std::ofstream &outfile) const
        {
            outfile << "<testcase ";
            outfile << "name=\"" << name_ << "\" ";
            outfile << "status=\"" << (run_? "run": "notrun") << "\"";
            outfile << ">" << std::endl;
            if (status_ != 0) {
                outfile << "<failure message=\"Exit code: " << status_ << "\"/>" << std::endl;
            }
            outfile << "</testcase>" << std::endl;
        }
    };

    std::string name_;
    std::vector<Variable> vars_;

    std::vector<TestCase> testCases_;

    void addTestCase(std::vector<VectorString::iterator> &current)
    {
        TestCase conf;

        for (size_t i = 0; i < current.size(); i++) {
            conf.addVariableInstance(vars_[i].getName(), *(current[i]));
        }
        testCases_.push_back(conf);
    }

    bool advance(std::vector<VectorString::iterator> &current,
                 std::vector<VectorString::iterator> &start,
                 std::vector<VectorString::iterator> &end)
    {
        if (equals(current, end)) return false;

        for (size_t i = 0; i < current.size(); i++) {
            if ((current[i] + 1) != end[i]) {
                current[i]++;
                break;
            } else {
                if (i != current.size() - 1) {
                    current[i] = start[i];
                }
            }
        }

        return true;
    }

    bool equals(std::vector<VectorString::iterator> &current,
                std::vector<VectorString::iterator> &end)
    {
        for (size_t i = 0; i < current.size(); i++) {
            if ((current[i] + 1) != end[i]) return false;
        }
        return true;
    }

    void generateTestCases()
    {
        std::vector<VectorString::iterator> start; 
        std::vector<VectorString::iterator> current; 
        std::vector<VectorString::iterator> end; 

        for (size_t i = 0; i < vars_.size(); i++) {
            start.push_back(vars_[i].begin());
            current.push_back(vars_[i].begin());
            end.push_back(vars_[i].end());
        }

        do {
            addTestCase(current);
        } while(advance(current, start, end));
    }
public:

    Test()
    {
    }

    Test(std::string name) :
        name_(name)
    {
    }

    Test &operator+=(Variable &v)
    {
        vars_.push_back(v);
        return *this;
    }

    std::string getName() const
    {
        return name_;
    }

    void launch()
    {
        generateTestCases();

        for (size_t i = 0; i < testCases_.size(); i++) {
            testCases_[i].run(name_);
        }
    }

    unsigned getNumberOfTestCases() const
    {
        return unsigned(testCases_.size());
    }

    unsigned getNumberOfFailures() const
    {
        unsigned failures = 0;
        for (size_t i = 0; i < testCases_.size(); i++) {
            if (testCases_[i].isFailure()) failures++; 
        }
        return failures;
    }

    void report(std::ofstream &outfile) const
    {
        outfile << "<testsuite ";
        outfile << "name=\"" << name_ << "\" ";
        outfile << "tests=\"" << getNumberOfTestCases() << "\" ";
        outfile << "failures=\"" << getNumberOfFailures() << "\" ";
        outfile << "errors=\"0\" ";
        outfile << ">" << std::endl;

        for (size_t i = 0; i < testCases_.size(); i++) {
            testCases_[i].report(outfile);
        }

        outfile << "</testsuite>" << std::endl;
    }
};

class TestSuite {
private:
    std::vector<Test> tests_;
    std::string name_;
    std::ofstream outfile_;

public:
    TestSuite(std::string name) :
        name_(name),
        outfile_((name + ".report.xml").c_str(), std::ofstream::out | std::ofstream::trunc)
    {
    }

    ~TestSuite()
    {
        outfile_.close();
    }

    TestSuite &operator+=(Test test)
    {
        tests_.push_back(test);
        return *this;
    }

    void launch()
    {
        for (size_t i = 0; i < tests_.size(); i++) {
            tests_[i].launch();
        }
    }

    unsigned getNumberOfTestCases() const
    {
        unsigned testCases = 0;
        for (size_t i = 0; i < tests_.size(); i++) {
            testCases += tests_[i].getNumberOfTestCases();
        }
        return testCases;
    }

    unsigned getNumberOfFailures() const
    {
        unsigned failures = 0;
        for (size_t i = 0; i < tests_.size(); i++) {
            failures += tests_[i].getNumberOfFailures();
        }
        return failures;
    }

    void report()
    {
        outfile_ << std::boolalpha;
        outfile_ << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>" << std::endl;

        outfile_ << "<testsuites ";
        outfile_ << "name=\"" << name_ << "\" ";
        outfile_ << "tests=\"" << getNumberOfTestCases() << "\" ";
        outfile_ << "failures=\"" << getNumberOfFailures() << "\" ";
        outfile_ << "errors=\"0\"";
        outfile_ << ">" << std::endl;

        for (size_t i = 0; i < tests_.size(); i++) {
            tests_[i].report(outfile_);
        }

        outfile_ << "</testsuites>" << std::endl;
    }
};

#endif /* TEST_H */

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
