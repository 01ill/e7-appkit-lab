#ifndef LPRTC_REGISTERS_HPP
#define LPRTC_REGISTERS_HPP


template <typename RegisterType, typename AddressType, AddressType Offset>
class Register {
    protected:
        volatile RegisterType* const reg;

    public:
        explicit Register(AddressType base) : reg(reinterpret_cast<volatile RegisterType*>(base + Offset)) { }
        
        /* Gibt den Inhalt des Registers zurück */
        RegisterType read() const { return *reg; }
        void inline write(RegisterType value) { *reg = value; }
        void inline set(RegisterType mask) { *reg |= mask; }
        void inline clear(RegisterType mask) { *reg &= ~mask; }
        bool is_set(RegisterType mask) const { return  (*reg & mask) != 0; }
};

#endif // LPRTC_REGISTERS_HPP