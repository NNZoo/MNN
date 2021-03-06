//
//  TFExtraManager.cpp
//  MNNConverter
//
//  Created by MNN on 2019/09/21.
//  Copyright © 2018, Alibaba Group Holding Limited
//

#include "TFExtraManager.hpp"
#include <mutex>
#include "MNN_generated.h"
namespace MNN {
namespace Express {
std::shared_ptr<TFExtraManager> TFExtraManager::gInstance;
static std::mutex gMutex;
std::shared_ptr<TFExtraManager> TFExtraManager::get() {
    std::unique_lock<std::mutex> _l(gMutex);
    if (nullptr == gInstance) {
        gInstance.reset(new TFExtraManager);
    }
    return gInstance;
}

void TFExtraManager::insert(const std::string& name, std::shared_ptr<Transform> transform) {
    mTransform.insert(std::make_pair(name, transform));
}
std::shared_ptr<TFExtraManager::Transform> TFExtraManager::find(const std::string& name) const {
    auto iter = mTransform.find(name);
    if (iter == mTransform.end()) {
        return nullptr;
    }
    return iter->second;
}


static auto gRegister = []() {
    auto extra = TFExtraManager::get();
    auto judge = [extra](VARP var) {
        auto op = var->expr().first->get();
        if (op->type() != OpType_Extra) {
            return false;
        }
        auto engine = op->main_as_Extra()->engine()->str();
        if (engine != "Tensorflow") {
            return false;
        }
        auto type = op->main_as_Extra()->type()->str();
        if (extra->find(type) == nullptr) {
            return false;
        }
        return true;
    };
    auto modify = [extra](VARP var) {
        auto op = var->expr().first->get();
        MNN_ASSERT(op->type() == OpType_Extra);
        auto type   = op->main_as_Extra()->type()->str();
        auto transformer = extra->find(type);
        MNN_ASSERT(nullptr != transformer);
        auto newExpr = transformer->onExecute(var->expr().first);
        if (nullptr == newExpr) {
            MNN_ERROR("Converte Tensorflow's Op %s , type = %s, failed, may be some node is not const\n", var->expr().first->name().c_str(), type.c_str());
            return false;
        }
        newExpr->setName(var->expr().first->name());
        auto outputs = var->expr().first->outputs();
        for (auto weakVar : outputs) {
            auto var = weakVar.lock();
            if (nullptr == var) {
                continue;
            }
            auto index = var->expr().second;
            Variable::setExpr(var, newExpr, index);
        }
        return true;
    };
    TemplateMerge::getInstance("TFExtra").insertTemplate("TFExtraManager", judge, modify);
    return true;
}();
}
}
