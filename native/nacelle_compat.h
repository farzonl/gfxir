#pragma once

#include <nacelle/nacelle.h>
#include <wrl/client.h>

#include <stdexcept>

inline void checkNacelleValidation(bool validation) {
    if (validation)
        throw std::runtime_error(
            "Nacelle v0.0.1 does not provide D3D12 validation (ID3D12InfoQueue); omit --validation");
}
