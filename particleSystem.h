#pragma once

#include <iostream>
#include <ostream>
#include <vector>
#include <algorithm>

#include <cuda_runtime.h>
#include <cufft.h>
#include <helper_math.h>

#include "types.h"
#include "randGenerator.h"

#define SQR(x) ((x)*(x))

template<typename T>
class ParticleSystem
{
    typedef std::vector<T> vecType;
    vecType vel;
    vecType nextVel;
    vecType pos;
    vecType prevPos;
    vecType nextPos;
    vecType force;
    std::vector<float3> color;
    std::vector<ParticleState> state;
    float mass;
    float radius;
    unsigned particles;
    unsigned step;

public:
    ParticleSystem(unsigned particlesAmount = 0)
        : mass(1.f),
        radius(0.03f),
        particles(particlesAmount),
        step(0u)
    {
        initialize();
        //initializeTest();
        std::cout << "ParticleSystem OK" << std::endl;
    }

    void initializeTest() {
        RandGenerator colorGenerator(256, 0);
        float v = 100.f;
        float a = 800.f; //nie moze byz za duze bo sie czastki kotluja przy zderzeniu np 800

        vel.push_back(T(v, 0.f));
        pos.push_back(T(-0.5f, 0.f));
        prevPos.push_back(pos[0]);
        force.push_back(float2(a, 0.f));

        vel.push_back(T(-v, 0.f));
        pos.push_back(T(0.5f, 0.f));
        prevPos.push_back(pos[1]);
        force.push_back(float2(-a, 0.f));

        vel.push_back(T(0.f, v));
        pos.push_back(T(0.f, -0.5f));
        prevPos.push_back(pos[2]);
        force.push_back(float2(0.f, a));

        vel.push_back(T(0.f, -v));
        pos.push_back(T(0.f, 0.5f));
        prevPos.push_back(pos[2]);
        force.push_back(float2(0.f, -a));

        for (unsigned i = 0; i < particles; i++) {
            nextPos.push_back(T(0.f, 0.f));
            nextVel.push_back(T(0.f, 0.f));
            state.push_back(inBox);

            if (particles == 4) {
                color.push_back(float3(1.f, 0.f, 0.f)); //red
                color.push_back(float3(0.f, 1.f, 0.f)); //green
                color.push_back(float3(1.f, 1.f, 0.f)); //yellow
                color.push_back(float3(0.5f, 0.f, 1.f)); //purpure
            }
            else {
                float3 col(colorGenerator.randValue(), colorGenerator.randValue(), colorGenerator.randValue());
                //std::cout << i << ") color: " << col << std::endl;
                color.push_back(col);
            }
        }
    }

    void initialize() {
        RandGenerator colorGenerator(256, 0);

        for (unsigned i = 0; i < particles; i++) {
            state.push_back(inEmiter);
            vel.push_back(T());
            nextVel.push_back(T());
            pos.push_back(T());
            prevPos.push_back(T());
            nextPos.push_back(T());
            force.push_back(T());

            float3 col = make_float3(colorGenerator.randValue(), colorGenerator.randValue(), colorGenerator.randValue());
            //std::cout << i << ") color: " << col << std::endl;
            color.push_back(col);
        }
    }

	void prepareMoveEuler(unsigned i, T acceleration, float dt) {
		nextVel[i] = vel[i] + acceleration * dt;
		nextPos[i] = pos[i] + nextVel[i] * dt;

		//std::cout << i << ") prepareMoveEuler - \t\tnextVel: " << nextVel[i] << " \tnexPos: " << nextPos[i] << std::en  dl;
	}


/*	void prepareMoveVerlet(unsigned i, T acceleration, float dt) {
		if(step == 0)
			prepareMoveEuler(i, acceleration, dt);

		nextPos[i] = pos[i] * 2.f - prevPos[i] + acceleration * SQR(dt);
		nextVel[i] = (nextPos[i] - prevPos[i]) * (1 / (2.f * dt));

		//std::cout << i << ") prepareMoveVerlet - \t\tnextVel: " << nextVel[i] << " \tnexPos: " << nextPos[i] << std::endl;
	}*/


	void prepareMove(unsigned i, float dt) {
        T acceleration = force[i] / mass;
        //std::cout << i << ") prepareMove - acceleration: " << acceleration  << " [" << force << " * " << 1/mass << "]"<< std::endl;

		//prepareMoveVerlet(i, acceleration, dt);
        prepareMoveEuler(i, acceleration, dt);
	}


	void move(unsigned i) {
		//std::cout << i << ") move - \t\t\tvel: " << vel[i] << " \t\tpos: " << pos[i] << std::endl;
		prevPos[i] = pos[i];
		pos[i] = nextPos[i];
		vel[i] = nextVel[i];	
	}

	void incrementStep() {
        std::cout << "STEP: " << step << std::endl << std::endl;
		step++;
	}










    void setParticlePos(unsigned i, T position) {
        pos[i] = position;
    }

    void setNextParticlePos(unsigned i, T position) {
        nextPos[i] = position;
    }

    void setPrevParticlePos(unsigned i, T position) {
        prevPos[i] = position;
    }

    void setParticleVel(unsigned i, T velocity) {
        vel[i] = velocity;
    }

    void setNextParticleVel(unsigned i, T velocity) {
        nextVel[i] = velocity;
    }

    void setParticleForce(unsigned i, T newForce) {
        force[i] = newForce;
    }

    void setParticleColor(unsigned i, float3 newColor) {
        color[i] = newColor;
    }

    void setParticleState(unsigned i, ParticleState newState) {
        state[i] = newState;
    }

    T getParticlePos(unsigned i) {
        return pos[i];
    }

    T getNextParticlePos(unsigned i) {
        return nextPos[i];
    }

    T getPrevParticlePos(unsigned i) {
        return prevPos[i];
    }

    T getParticleVel(unsigned i) {
        return vel[i];
    }

    T getNextParticleVel(unsigned i) {
        return nextVel[i];
    }

    T getForce(unsigned i) {
        return force[i];
    }

    unsigned getParticleAmount() {
        return particles;
    }

    float3 getParticleColor(unsigned i) {
        return color[i];
    }

    ParticleState getParticleState(unsigned i) {
        return state[i];
    }
};
