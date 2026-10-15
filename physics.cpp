#include "menu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

struct Vec2 {
    double x = 0, y = 0;
    Vec2() = default;
    Vec2(double x_, double y_) : x(x_), y(y_) {}
    Vec2 operator+(const Vec2& o) const { return {x + o.x, y + o.y}; }
    Vec2 operator-(const Vec2& o) const { return {x - o.x, y - o.y}; }
    Vec2 operator*(double s) const { return {x * s, y * s}; }
    Vec2& operator+=(const Vec2& o) { x += o.x; y += o.y; return *this; }
    Vec2& operator-=(const Vec2& o) { x -= o.x; y -= o.y; return *this; }
    double dot(const Vec2& o) const { return x * o.x + y * o.y; }
    double length() const { return std::sqrt(x * x + y * y); }
};

struct Body {
    Vec2 pos, vel, force;
    double radius = 1.0;
    double mass = 1.0;
    double invMass = 1.0;   // 0 = immovable
    double restitution = 0.8;
    char glyph = 'o';

    void setMass(double m) {
        mass = m;
        invMass = (m > 0) ? 1.0 / m : 0.0;
    }
    double kinetic() const { return 0.5 * mass * vel.dot(vel); }
};

class World {
public:
    double width, height;
    Vec2 gravity{0, -9.81};
    double drag = 0.0;              // linear air drag coefficient
    std::vector<Body> bodies;

    World(double w, double h) : width(w), height(h) {}

    void add(const Body& b) { bodies.push_back(b); }

    void step(double dt, int substeps = 4) {
        double h = dt / substeps;
        for (int i = 0; i < substeps; ++i) {
            applyForces();
            integrate(h);
            for (int pass = 0; pass < 2; ++pass) solveCollisions();
            solveWalls();
        }
    }

    double totalEnergy() const {
        double e = 0;
        for (const auto& b : bodies) {
            e += b.kinetic();
            e += b.mass * -gravity.y * b.pos.y;   // potential energy
        }
        return e;
    }

private:
    void applyForces() {
        for (auto& b : bodies) {
            if (b.invMass == 0) continue;
            b.force = gravity * b.mass;
            b.force -= b.vel * drag;
        }
    }

    // Semi-implicit (symplectic) Euler: update velocity first, then position.
    void integrate(double h) {
        for (auto& b : bodies) {
            if (b.invMass == 0) continue;
            b.vel += b.force * (b.invMass * h);
            b.pos += b.vel * h;
        }
    }

    void solveWalls() {
        for (auto& b : bodies) {
            if (b.pos.x - b.radius < 0)      { b.pos.x = b.radius;          if (b.vel.x < 0) b.vel.x = -b.vel.x * b.restitution; }
            if (b.pos.x + b.radius > width)  { b.pos.x = width - b.radius;  if (b.vel.x > 0) b.vel.x = -b.vel.x * b.restitution; }
            if (b.pos.y - b.radius < 0)      { b.pos.y = b.radius;          if (b.vel.y < 0) b.vel.y = -b.vel.y * b.restitution; }
            if (b.pos.y + b.radius > height) { b.pos.y = height - b.radius; if (b.vel.y > 0) b.vel.y = -b.vel.y * b.restitution; }
        }
    }

    void solveCollisions() {
        for (size_t i = 0; i < bodies.size(); ++i) {
            for (size_t j = i + 1; j < bodies.size(); ++j) {
                Body& a = bodies[i];
                Body& b = bodies[j];

                Vec2 d = b.pos - a.pos;
                double minDist = a.radius + b.radius;
                double distSq = d.dot(d);
                if (distSq >= minDist * minDist) continue;

                double dist = std::sqrt(distSq);
                Vec2 n = (dist > 1e-9) ? d * (1.0 / dist) : Vec2(1, 0);
                double penetration = minDist - dist;

                double invSum = a.invMass + b.invMass;
                if (invSum == 0) continue;

                // Positional correction (push apart, weighted by inverse mass)
                const double percent = 0.8, slop = 0.001;
                Vec2 correction = n * (std::max(penetration - slop, 0.0) / invSum * percent);
                a.pos -= correction * a.invMass;
                b.pos += correction * b.invMass;

                // Impulse along the contact normal
                Vec2 rv = b.vel - a.vel;
                double vn = rv.dot(n);
                if (vn > 0) continue;   // already separating

                double e = std::min(a.restitution, b.restitution);
                double j_ = -(1.0 + e) * vn / invSum;
                Vec2 impulse = n * j_;
                a.vel -= impulse * a.invMass;
                b.vel += impulse * b.invMass;
            }
        }
    }
};

// ---------- ASCII renderer ----------
void render(const World& w, int cols, int rows, double fps) {
    std::vector<std::string> grid(rows, std::string(cols, ' '));
    double cw = w.width / cols;
    double ch = w.height / rows;

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            Vec2 p((c + 0.5) * cw, w.height - (r + 0.5) * ch);
            for (const auto& b : w.bodies) {
                Vec2 d = p - b.pos;
                if (d.dot(d) <= b.radius * b.radius) { grid[r][c] = b.glyph; break; }
            }
        }
    }

    std::string out = "\x1b[H";   // cursor home
    out += "+" + std::string(cols, '-') + "+\n";
    for (const auto& row : grid) out += "|" + row + "|\n";
    out += "+" + std::string(cols, '-') + "+\n";
    char info[128];
    std::snprintf(info, sizeof info, "bodies: %zu  energy: %9.2f  fps: %5.1f   \n",
                  w.bodies.size(), w.totalEnergy(), fps);
    out += info;
    std::fputs(out.c_str(), stdout);
    std::fflush(stdout);
}

int main() {
    const int COLS = 80, ROWS = 24;
    const double WORLD_W = 40.0, WORLD_H = 24.0 * 0.5 * 2;   // keeps cells ~square-looking
    World world(WORLD_W, WORLD_H);
    world.drag = 0.02;

    std::srand(7);
    auto rnd = [](double lo, double hi) { return lo + (hi - lo) * (std::rand() / (double)RAND_MAX); };

    const char glyphs[] = "oO@#*%&";
    for (int i = 0; i < 14; ++i) {
        Body b;
        b.radius = rnd(0.8, 1.8);
        b.setMass(3.14159 * b.radius * b.radius);    // mass ~ area
        b.pos = {rnd(3, WORLD_W - 3), rnd(8, WORLD_H - 2)};
        b.vel = {rnd(-8, 8), rnd(-4, 4)};
        b.restitution = 0.85;
        b.glyph = glyphs[i % 7];
        world.add(b);
    }

    std::printf("\x1b[2J\x1b[?25l");   // clear screen, hide cursor

    using clock = std::chrono::steady_clock;
    const double dt = 1.0 / 60.0;
    double fps = 60;
    auto last = clock::now();

    while (true) {
        world.step(dt);
        render(world, COLS, ROWS, fps);

        std::this_thread::sleep_for(std::chrono::duration<double>(dt));
        auto now = clock::now();
        double frame = std::chrono::duration<double>(now - last).count();
        fps = 0.9 * fps + 0.1 * (1.0 / frame);
        last = now;
    }
}