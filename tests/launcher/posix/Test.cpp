#include <sys/types.h>
#include <sys/wait.h>

#include <cassert>
#include <iostream>
#include <fstream>

#include "../Test.h"
    
Test::TestCase::Stats
Test::TestCase::run(const std::string &exec)
{
    Stats stats;

    std::cout << "Launching: "<< exec << " - " << name_ << std::endl;
    int fd[2];
    int ret = pipe(fd);
    assert(ret == 0);

    gmactime_t start, end;

    ::getTime(&start);
    pid_t pid = fork();
    if (pid == 0) {
        setEnvironment();
        close(fd[0]);
        ret = dup2(fd[1], 1);
        assert(ret == 1);
        int execReturn = execlp(exec.c_str(), exec.c_str(), (char *)NULL);
        printf("Failure! execve error code %d\n", execReturn);
        abort();
    }
    else if(pid < 0) {
        printf("Failure! failed to fork %d\n", pid);
        abort();
    } else {
        close(fd[1]);
        run_ = true;
        char *buf = NULL;
        ssize_t bytes;
        size_t bytes2 = 4096;
        FILE *file = fdopen(fd[0], "r");
        assert(file != NULL);

        while ((bytes = getline(&buf, &bytes2, file)) >= 0) {
            if (bytes == 0) continue;
            buf[bytes - 1] = '\0';
            std::string line(buf);
            // Get the name of the field
            std::istringstream iss(line);
            std::string name;
            std::getline(iss, name, ':');

            // Get the value of the field
            std::string token;

            while (std::getline(iss, token)) {
                float val;
                //line.erase(remove_if(line.begin(), line.end(), isspace), line.end());
                std::istringstream convert(token);
                convert >> std::dec >> val;
                stats.addPair(name, val);
            }

            free(buf);
            buf = NULL;
        }

        ::wait(&status_); 
        ::getTime(&end);
        setElapsedTime(((end.sec - start.sec) * 1000000 + (end.usec - start.usec)) / 1000);
        fclose(file);
    }
    return stats;
}

void
Test::TestCase::setEnvironment()
{
    std::vector<KeyVal>::const_iterator it;
    for (it = keyvals_.begin(); it != keyvals_.end(); it++) {
        ::setenv(it->first.c_str(), it->second.c_str(), 1);
    }
    ::setenv("PATH", ".", 1);
}

/* vim:set backspace=2 tabstop=4 shiftwidth=4 textwidth=120 foldmethod=marker expandtab: */
