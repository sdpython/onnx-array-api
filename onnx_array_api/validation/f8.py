import struct
import numpy

# display functions


def display_float32(value, sign=1, exponent=8, mantissa=23):
    """
    Displays a float32 into b.

    :param value: value to display (float32)
    :param sign: number of bits for the sign
    :param exponent: number of bits for the exponent
    :param mantissa: number of bits for the mantissa
    :return: string
    """
    t = sign + exponent + mantissa
    ival = int.from_bytes(struct.pack("<f", numpy.float32(value)), "little")
    s = bin(ival)[2:]
    s = "0" * (t - len(s)) + s
    s1 = s[:sign]
    s2 = s[sign : sign + exponent]
    s3 = s[sign + exponent :]
    return ".".join([s1, s2, s3])


def display_float16(value, sign=1, exponent=5, mantissa=10):
    """
    Displays a float32 into b.

    :param value: value to display (float16)
    :param sign: number of bits for the sign
    :param exponent: number of bits for the exponent
    :param mantissa: number of bits for the mantissa
    :return: string
    """
    t = sign + exponent + mantissa
    ival = numpy.float16(value).view("H")  # pylint: disable=E1121
    s = bin(ival)[2:]
    s = "0" * (t - len(s)) + s
    s1 = s[:sign]
    s2 = s[sign : sign + exponent]
    s3 = s[sign + exponent :]
    return ".".join([s1, s2, s3])


def display_fexmx(value, sign, exponent, mantissa):
    """
    Displays any float encoded with 1 bit for the sign,
    *exponent* bit for the exponent and *mantissa* bit for the
    mantissa.

    :param value: value to display (int)
    :param sign: number of bits for the sign
    :param exponent: number of bits for the exponent
    :param mantissa: number of bits for the mantissa
    :return: string
    """
    t = sign + exponent + mantissa
    ival = value
    s = bin(ival)[2:]
    s = "0" * (t - len(s)) + s
    s1 = s[:sign]
    s2 = s[sign : sign + exponent]
    s3 = s[sign + exponent :]
    return ".".join([s1, s2, s3])


def display_fe4m3(value, sign=1, exponent=4, mantissa=3):
    """
    Displays a float 8 E4M3 into b.

    :param value: value to display (int)
    :param sign: number of bits for the sign
    :param exponent: number of bits for the exponent
    :param mantissa: number of bits for the mantissa
    :return: string
    """
    return display_fexmx(value, sign=1, exponent=4, mantissa=3)


def display_fe5m2(value, sign=1, exponent=4, mantissa=3):
    """
    Displays a float 8 E5M2 into binary format.

    :param value: value to display (int)
    :param sign: number of bits for the sign
    :param exponent: number of bits for the exponent
    :param mantissa: number of bits for the mantissa
    :return: string
    """
    return display_fexmx(value, sign=1, exponent=5, mantissa=2)


# cast from float 8 to float 32


def fe4m3_to_float32_float(ival: int, fn: bool = True, uz: bool = False) -> float:
    """
    Casts a float 8 encoded as an integer into a float.

    :param ival: byte
    :param fn: no infinite values
    :param uz: no negative zero
    :return: float (float 32)
    """
    if not fn:
        raise NotImplementedError("fn=False is not implemented.")
    if ival < 0 or ival > 255:
        raise ValueError(f"{ival} is not a float8.")
    if ival == 0:
        return numpy.float32(0)
    if uz:
        if ival == 0x80:
            return numpy.float32(numpy.nan)
        sign = ival & 0x80
        ival &= 0x7F
        expo = ival >> 3
        mant = ival & 0x07
        powe = expo & 0x0F
        if expo == 0:
            powe -= 7
            fraction = 0
        else:
            powe -= 8
            fraction = 1
        fval = float(mant / 8 + fraction) * 2.0**powe
        if sign:
            fval = -fval
        return numpy.float32(fval)
    else:
        if ival == 255:
            return numpy.float32(-numpy.nan)
        if ival == 127:
            return numpy.float32(numpy.nan)
        sign = ival & 0x80
        if ival == 0 and sign > 0:
            return -numpy.float32(0)

        ival &= 0x7F
        expo = ival >> 3
        mant = ival & 0x07
        powe = expo & 0x0F
        if expo == 0:
            powe -= 6
            fraction = 0
        else:
            powe -= 7
            fraction = 1
        fval = float(mant / 8 + fraction) * 2.0**powe
        if sign:
            fval = -fval
        return numpy.float32(fval)


def fe5m2_to_float32_float(ival: int, fn: bool = False, uz: bool = False) -> float:
    """
    Casts a float 8 encoded as an integer into a float.

    :param ival: byte
    :param fn: no infinite values
    :param uz: no negative zero
    :return: float (float 32)
    """
    if ival < 0 or ival > 255:
        raise ValueError(f"{ival} is not a float8.")
    if fn and uz:
        if ival == 0x80:
            return numpy.float32(numpy.nan)
        exponent_bias = 16
    elif not fn and not uz:
        if ival in (255, 254, 253):
            return numpy.float32(-numpy.nan)
        if ival in (127, 126, 125):
            return numpy.float32(numpy.nan)
        if ival == 252:
            return -numpy.float32(numpy.inf)
        if ival == 124:
            return numpy.float32(numpy.inf)
        if (ival & 0x7F) == 0:
            return numpy.float32(0)
        exponent_bias = 15
    else:
        raise NotImplementedError("fn and uz must be both True or False.")

    sign = ival & 0x80
    ival &= 0x7F
    expo = ival >> 2
    mant = ival & 0x03
    powe = expo & 0x1F
    if expo == 0:
        powe -= exponent_bias - 1
        fraction = 0
    else:
        powe -= exponent_bias
        fraction = 1
    fval = float(mant / 4 + fraction) * 2.0**powe
    if sign:
        fval = -fval
    return numpy.float32(fval)


def fe4m3_to_float32(ival: int, fn: bool = True, uz: bool = False) -> float:
    """
    Casts a float E4M3 encoded as an integer into a float.

    :param ival: byte
    :param fn: no inifinite values
    :param uz: no negative zero
    :return: float (float 32)
    """
    if not fn:
        raise NotImplementedError("fn=False is not implemented.")
    if ival < 0 or ival > 255:
        raise ValueError(f"{ival} is not a float8.")
    if uz:
        exponent_bias = 8
        if ival == 0x80:
            return numpy.nan
    else:
        exponent_bias = 7
        if ival == 255:
            return numpy.float32(-numpy.nan)
        if ival == 127:
            return numpy.float32(numpy.nan)

    expo = (ival & 0x78) >> 3
    mant = ival & 0x07
    sign = ival & 0x80
    res = sign << 24
    if expo == 0:
        if mant > 0:
            expo = 0x7F - exponent_bias
            if mant & 0x4 == 0:
                mant &= 0x3
                mant <<= 1
                expo -= 1
            if mant & 0x4 == 0:
                mant &= 0x3
                mant <<= 1
                expo -= 1
            res |= (mant & 0x3) << 21
            res |= expo << 23
    else:
        res |= mant << 20
        expo += 0x7F - exponent_bias
        res |= expo << 23
    f = numpy.uint32(res).view(numpy.float32)  # pylint: disable=E1121
    return f


def fe5m2_to_float32(ival: int, fn: bool = False, uz: bool = False) -> float:
    """
    Casts a float E5M2 encoded as an integer into a float.

    :param ival: byte
    :param fn: no inifinite values
    :param uz: no negative values
    :return: float (float 32)
    """
    if ival < 0 or ival > 255:
        raise ValueError(f"{ival} is not a float8.")
    if fn and uz:
        if ival == 0x80:
            return numpy.float32(numpy.nan)
        exponent_bias = 16
    elif not fn and not uz:
        if ival in {253, 254, 255}:
            return numpy.float32(-numpy.nan)
        if ival in {125, 126, 127}:
            return numpy.float32(numpy.nan)
        if ival == 252:
            return numpy.float32(-numpy.inf)
        if ival == 124:
            return numpy.float32(numpy.inf)
        exponent_bias = 15
    else:
        raise NotImplementedError("fn and uz must be both False or True.")

    expo = (ival & 0x7C) >> 2
    mant = ival & 0x03
    sign = ival & 0x80
    res = sign << 24
    if expo == 0:
        if mant > 0:
            expo = 0x7F - exponent_bias
            if mant & 0x2 == 0:
                mant &= 0x1
                mant <<= 1
                expo -= 1
            res |= (mant & 0x1) << 22
            res |= expo << 23
    else:
        res |= mant << 21
        expo += 0x7F - exponent_bias
        res |= expo << 23
    f = numpy.uint32(res).view(numpy.float32)  # pylint: disable=E1121
    return f


# cast from float32 to float 8


class CastFloat8:
    """
    Helpers to cast float8 into float32 or the other way around.
    """

    values_e4m3fn = list(
        sorted(
            (fe4m3_to_float32_float(i), i) for i in range(0, 256) if i not in (255, 127)
        )
    )

    values_e4m3fnuz = list(
        sorted(
            (fe4m3_to_float32_float(i, uz=True), i) for i in range(0, 256) if i != 0x80
        )
    )

    values_e5m2 = list(
        sorted(
            (fe5m2_to_float32_float(i), i)
            for i in range(0, 256)
            if i not in {253, 254, 255, 125, 126, 127}
        )
    )

    values_e5m2fnuz = list(
        sorted(
            (fe5m2_to_float32_float(i, fn=True, uz=True), i)
            for i in range(0, 256)
            if i != 0x80
        )
    )

    @staticmethod
    def find_closest_value(value, sorted_values):
        """
        Search a value into a sorted array of values.

        :param value: float32 value to search
        :param sorted_values: list of tuple `[(float 32, byte)]`
        :return: byte

        The function searches into the first column the closest value and
        return the value on the second columns.
        """
        a = 0
        b = len(sorted_values)
        while a < b:
            m = (a + b) // 2
            th = sorted_values[m][0]
            if value == th:
                return sorted_values[m][1]
            if value < th:
                b = m
            elif a == m:
                break
            else:
                a = m
        # finds the closest one
        if b < len(sorted_values):
            d1 = value - sorted_values[a][0]
            d2 = sorted_values[b][0] - value
            if d1 < d2:
                return sorted_values[a][1]
            if d1 == d2:
                return sorted_values[a][1] if value < 0 else sorted_values[b][1]
            return sorted_values[b][1]
        return sorted_values[a][1]


def search_float32_into_fe4m3(value: float, fn: bool = True, uz: bool = False) -> int:
    """
    Casts a float 32 into a float E4M3.

    :param value: float
    :param fn: no infinite values
    :param uz: no negative zero
    :return: byte
    """
    if not fn:
        raise NotImplementedError("fn=False is not implemented.")

    b = int.from_bytes(struct.pack("<f", numpy.float32(value)), "little")
    ret = (b & 0x80000000) >> 24  # sign
    if uz:
        if numpy.isnan(value) or numpy.isinf(value):
            return 0x80
        set_values = CastFloat8.values_e4m3fnuz
    else:
        if numpy.isnan(value) or numpy.isinf(value):
            return 0x7F | ret
        set_values = CastFloat8.values_e4m3fn
    f = numpy.float32(value)
    i = CastFloat8.find_closest_value(f, set_values)
    return (i & 0x7F) | ret


def search_float32_into_fe5m2(value: float, fn: bool = False, uz: bool = False) -> int:
    """
    Casts a float 32 into a float E5M2.

    :param value: float
    :param fn: no infinite values
    :param uz: no negative zero
    :return: byte
    """
    b = int.from_bytes(struct.pack("<f", numpy.float32(value)), "little")
    ret = (b & 0x80000000) >> 24  # sign

    if fn and uz:
        if numpy.isnan(value) or numpy.isinf(value):
            return 0x80
        set_values = CastFloat8.values_e5m2fnuz
    elif not fn and not uz:
        if numpy.isnan(value):
            return 0x7F | ret
        set_values = CastFloat8.values_e5m2
    else:
        raise NotImplementedError("fn and uz must both True or False.")

    f = numpy.float32(value)
    i = CastFloat8.find_closest_value(f, set_values)
    return (i & 0x7F) | ret


def float32_to_fe4m3(x, fn: bool = True, uz: bool = False):
    """
    Converts a float32 into a float E4M3.

    :param x: numpy.float32
    :param fn: no infinite values
    :param uz: no negative zero
    :return: byte
    """
    if not fn:
        raise NotImplementedError("fn=False is not implemented.")
    b = int.from_bytes(struct.pack("<f", numpy.float32(x)), "little")
    ret = (b & 0x80000000) >> 24  # sign
    if uz:
        if (b & 0x7FC00000) == 0x7FC00000 or numpy.isinf(x):
            return 0x80
        e = (b & 0x7F800000) >> 23  # exponent
        m = b & 0x007FFFFF  # mantissa

        if e != 0:
            if e < 116:
                pass
            elif e < 117:
                ret |= 1
                if (m >> 23) & 1:
                    # rounding
                    ret += 1
            elif e < 120:  # 127 - 8 + 1
                d = 119 - e
                ret |= 1 << (2 - d)
                ret |= m >> (21 + d)
                if (m >> (20 + d)) & 1:
                    # rounding
                    ret += 1
            elif e < 135:  # 127 + 8
                ex = e - 119  # 127 - 8
                if ex == 0:
                    ret |= 0x4
                    ret |= m >> 21
                else:
                    ret |= ex << 3
                    ret |= m >> 20
                if (m & 0x80000) and (ret & 0x7F) < 0x7F:
                    # rounding
                    ret += 1
            else:
                ret |= 0x7F  # 01111110
        return int(ret)
    else:
        if (b & 0x7FC00000) == 0x7FC00000 or numpy.isinf(x):
            return 0x7F | ret
        e = (b & 0x7F800000) >> 23  # exponent
        m = b & 0x007FFFFF  # mantissa

        if e != 0:
            if e < 117:
                pass
            elif e < 118:
                ret |= 1
                if (m >> 23) & 1:
                    # rounding
                    ret += 1
            elif e < 121:  # 127 - 7 + 1
                d = 120 - e
                ret |= 1 << (2 - d)
                ret |= m >> (21 + d)
                if (m >> (20 + d)) & 1:
                    # rounding
                    ret += 1
            elif e < 136:  # 127 + 8 + 1
                ex = e - 120  # 127 - 7
                if ex == 0:
                    ret |= 0x4
                    ret |= m >> 21
                else:
                    ret |= ex << 3
                    ret |= m >> 20
                    if (ret & 0x7F) == 0x7F:
                        ret &= 0xFE
                if (m & 0x80000) and (ret & 0x7F) < 0x7E:
                    # rounding
                    ret += 1
            else:
                ret |= 126  # 01111110
        return int(ret)


def float32_to_fe5m2(x, fn: bool = False, uz: bool = False):
    """
    Converts a float32 into a float E5M2.

    :param x: numpy.float32
    :param fn: no infinite values
    :param uz: no negative zero
    :return: byte
    """
    b = int.from_bytes(struct.pack("<f", numpy.float32(x)), "little")
    ret = (b & 0x80000000) >> 24  # sign

    if fn and uz:
        if (b & 0x7FC00000) == 0x7FC00000:  # NaN
            return 0x80
        if (b & 0x7FFFFFFF) == 0x7F800000:  # Inf
            return 0x80
        e = (b & 0x7F800000) >> 23  # exponent
        m = b & 0x007FFFFF  # mantissa

        if e != 0:
            if e < 109:
                pass
            elif e < 110:
                ret |= 1
                if (m >> 23) & 1:
                    # rounding
                    # may be unused
                    ret += 1
            elif e < 112:  # 127 - 16 + 1
                d = 111 - e
                ret |= 1 << (1 - d)
                ret |= m >> (22 + d)
                if (m >> (21 + d)) & 1:
                    # rounding
                    ret += 1
            elif e < 143:  # 127 + 15 + 1
                ex = e - 111  # 127 - 16
                ret |= ex << 2
                ret |= m >> 21
                if (m & 0x100000) and (ret & 0x7F) < 0x7F:
                    # rounding
                    ret += 1
            elif e == 255 and m == 0:  # inf
                return 0x80
            else:
                ret |= 0x7F  # last possible number
        return int(ret)
    elif not fn and not uz:
        if (b & 0x7FC00000) == 0x7FC00000:
            return 0x7F | ret
        e = (b & 0x7F800000) >> 23  # exponent
        m = b & 0x007FFFFF  # mantissa

        if e != 0:
            if e < 110:
                pass
            elif e < 111:
                ret |= 1
                if (m >> 23) & 1:
                    # rounding
                    # may be unused
                    ret += 1
            elif e < 113:  # 127 - 15 + 1
                d = 112 - e
                ret |= 1 << (1 - d)
                ret |= m >> (22 + d)
                if (m >> (21 + d)) & 1:
                    # rounding
                    ret += 1
            elif e < 143:  # 127 + 15 + 1
                ex = e - 112  # 127 - 15
                ret |= ex << 2
                ret |= m >> 21
                if (m & 0x100000) and (ret & 0x7F) < 0x7B:
                    # rounding
                    ret += 1
            elif e == 255 and m == 0:  # inf
                ret |= 124
            else:
                ret |= 123
        return int(ret)
    else:
        raise NotImplementedError("fn and uz must be both False or True.")
