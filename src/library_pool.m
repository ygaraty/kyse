// A fresh autorelease pool for each native call from Python's inference thread.
#import <Foundation/Foundation.h>
int kyse_with_pool(int (*operation)(void *), void *argument) {
    @autoreleasepool { return operation(argument); }
}
