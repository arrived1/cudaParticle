#pragma once

#include "particleSystem.h"


template <class ParticleSystem, typename T>
class Emiter
{
    ParticleSystem& pSystem;
    const T position;
    RandGenerator *velGeneratorX, *velGeneratorY, *forceGenerator;
public:

    Emiter(ParticleSystem& system, const T pos)
        : pSystem(system),
        position(pos)
    {
        int wingRadius = 3;
        velGeneratorX = new RandGenerator(100, 0);
        velGeneratorY = new RandGenerator(3, 0); //3, -3
        forceGenerator = new RandGenerator(100, 0);
    }

    void emit(unsigned i) {
        if (pSystem.getParticleState(i) == inEmiter) {
            pSystem.setParticleState(i, inSafeState);
            
            pSystem.setParticlePos(i, position);
            pSystem.setPrevParticlePos(i, position);

            float2 vel = make_float2(velGeneratorX->randValue(), velGeneratorY->randValue());
            pSystem.setParticleVel(i, vel);
            pSystem.setParticleForce(i, make_float2(forceGenerator->randValue(), 0.f));
        }
    }

    void backToEmiter(unsigned i) {
        pSystem.setParticleState(i, inEmiter);
        pSystem.setParticlePos(i, position);
        pSystem.setPrevParticlePos(i, position);
        pSystem.setNextParticlePos(i, T());
        pSystem.setParticleVel(i, T());
        pSystem.setNextParticleVel(i, T());
        pSystem.setParticleForce(i, T());
    }

    T getPosition() {
        return position;
    }
};