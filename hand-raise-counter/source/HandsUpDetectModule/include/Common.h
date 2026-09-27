//
// Created by kun on 2022/3/26.
//

#ifndef HandsUpVOTING_COMMON_H
#define HandsUpVOTING_COMMON_H

namespace smartmore
{
    //!
    //! \enum ResultCode
    //!
    enum class ResultCode
    {
        Success = 0,

        NotInited = 10001,
        BadModel = 10101,
        BadInput = 10102,
        NotImage = 10103,
        NotVideo = 10104,

        InternalError = 20001,
        OutOfMemory = 20002,
        CudaError = 20003,

        NoLicense = 30000,
        CallLimited = 30001,
        LicenseExpired = 30002,
        WrongAppName = 30003,
    };

    //!
    //! \struct BBox coordinate and confidence of detected object
    //! xmin, ymin, xmax, ymax : coordinates of box
    //! score : confidence of box
    //!
    struct BBox {
        float xmin, ymin, xmax, ymax, score;
    };

}

#endif //HandsUpVOTING_COMMON_H
